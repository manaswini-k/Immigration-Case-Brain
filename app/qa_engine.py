from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import faiss
import docx
import spacy
import os
from pathlib import Path
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
import re
from typing import Optional, List

DATA_PATH = Path("data")
DATA_PATH.mkdir(parents=True, exist_ok=True)

CASES: dict[str, dict] = {}  


_RFE_TITLES = (
    "Educational mismatch", "Generic job title", "Specialty occupation doubt",
    "In-house employment", "Degree/education evidence", "Employer–employee relationship"
)


OFFTOPIC_MIN_SCORE = 0.32


def _best_score(items: List["RetrievedItem"]) -> float:
    return max((it.score for it in items), default=0.0)



def _extract_rfe_concern(text: str) -> Optional[str]:
    """
    Extract the main RFE concern from text in a robust way.
    Handles multiple known formats and falls back to sniffing known titles.
    """
    t = " ".join(text.split())

    m = re.search(r"Subject:\s*Response to RFE\s*[-–—]\s*([A-Za-z ][A-Za-z \-/]{2,100})", t, re.IGNORECASE)
    if m:
        concern = m.group(1).strip().rstrip(" .")
    else:
    
        m = re.search(r"\bConcern\s*:\s*([A-Za-z ][A-Za-z \-/]{2,100})", t, re.IGNORECASE)
        if m:
            concern = m.group(1).strip().rstrip(" .")
        else:
      
            for title in _RFE_TITLES:
                if re.search(re.escape(title), t, re.IGNORECASE):
                    concern = title
                    break
            else:
                return None


    concern = re.sub(r"\bTo Whom It May Concern\b.*$", "", concern, flags=re.IGNORECASE).strip(" .")
    return concern or None




REG_TRIGGER_WORDS = {
    "regulation","regulations","clause","section","statute",
    "8 cfr","cfr","code of federal regulations","ina","§","section","214.2","214"
}


REGEX_REGULATION = re.compile(
    r"""(?ix)
    (?:\b(?:8\s*\.?\s*c\.?f\.?r\.?|cfr)\b\s*[§]?\s*[\d\.]+[\w\(\)\.-]*)
    |(?:\bina\b\s*[§]\s*[\d\.]+[\w\(\)\.-]*)
    |(?:[§]\s*[\d\.]+[\w\(\)\.-]*)
    """,
)


APPROVED_TOKENS = {
    "approved", "approval notice", "i-797 approval", "petition approved", "h-1b approved"
}
DENIED_TOKENS = {
    "denied", "petition denied", "decision: denied", "notice of denial"
}
RFE_TOKENS = {
    "request for evidence", "rfe", "r.f.e."
}

def _detect_outcome(text: str) -> Optional[str]:

    t = text.lower()
   
    if any(tok in t for tok in DENIED_TOKENS):
        return "denied"
    if any(tok in t for tok in APPROVED_TOKENS):
        return "approved"
    if any(tok in t for tok in RFE_TOKENS):
        return "rfe"
    return None

nlp = spacy.load("en_core_web_sm")
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


model = SentenceTransformer("all-MiniLM-L6-v2")


documents = []
embeddings = []

def is_regulation_query(q: str) -> bool:
    qn = q.lower()
    return any(t in qn for t in REG_TRIGGER_WORDS)

def filter_chunks_for_regulations(chunks: list[str]) -> list[str]:
    kept = []
    for ch in chunks:
        if REGEX_REGULATION.search(ch):
            kept.append(ch)
    return kept

def extract_reg_citations(*texts: str) -> list[str]:
    found = []
    for t in texts:
        if not t:
            continue
        for m in REGEX_REGULATION.finditer(t):
            citation = normalize_citation(m.group(0))
            if citation:
                found.append(citation)

    seen = set()
    uniq = []
    for c in found:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def normalize_citation(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw.strip())
   
    s = s.replace("C.F.R.", "CFR").replace("cfr", "CFR").replace("Cfr", "CFR")
  
    s = s.replace("§ ", "§ ").replace("§", " § ")
    s = re.sub(r"\s{2,}", " ", s).strip()
  
    s = s.replace(" § §", " §")
    return s

def _case_touch(fn: str):
    if fn not in CASES:
        CASES[fn] = {
            "petitioner": None, "beneficiary": None,
            "receipt_number": None, "receipt_date": None,
            "decision_date": None, "regulations": set(),
            "type": None
        }

def _case_absorb(fn: str, text: str):
    _case_touch(fn)
 
    t = _detect_outcome(text)  # "denied"|"approved"|"rfe"|None
    if t and not CASES[fn]["type"]:
        CASES[fn]["type"] = {"denied":"denial","approved":"approval","rfe":"rfe"}[t]
  
    for k in ("petitioner","beneficiary","receipt_number","receipt_date","decision_date"):
        v = _extract_fact_from_text(k, text) if k in FACT_PATTERNS else None
        if v and not CASES[fn][k]:
            CASES[fn][k] = v
 
    for r in extract_regulations(text):
        CASES[fn]["regulations"].add(normalize_citation(r))


def load_documents():
    if not DATA_PATH.exists():
        print("❌ Data folder not found:", DATA_PATH)
        return

    for file in DATA_PATH.iterdir():
        if file.suffix.lower() == ".pdf":
            reader = PdfReader(str(file))
            page_texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    page_texts.append(text)
                    chunks = split_text(text)
                    for chunk in chunks:
                        documents.append((file.name, chunk))
                        embeddings.append(model.encode(chunk))
            if page_texts:
                _case_absorb(file.name, " ".join(page_texts))

        elif file.suffix.lower() == ".docx":
            doc = docx.Document(str(file))
            full_text = "\n".join([p.text for p in doc.paragraphs])
            if full_text.strip():
                chunks = split_text(full_text)
                for chunk in chunks:
                    documents.append((file.name, chunk))
                    embeddings.append(model.encode(chunk))
                _case_absorb(file.name, full_text)


def split_text(text, max_tokens=80):
    """
    Split text into clean, meaningful chunks using SpaCy sentence segmentation and chunk grouping.
    Each chunk contains up to ~80 tokens, grouped by sentences.
    """
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    current_len = 0

    for sent in doc.sents:
        sentence = sent.text.strip().replace("\n", " ")
        sent_len = len(sentence.split())

      
        if sent_len < 5:
            continue

       
        if current_len + sent_len > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_len = sent_len
        else:
            current_chunk += " " + sentence
            current_len += sent_len

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def build_faiss_index():
    if not embeddings:
        return None
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def _mentions_any_entity(text: str, names: List[str]) -> bool:
    """
    Return True if the text contains any of the provided names.
    Each name may have multiple words; we require all parts to be present.
    """
    t = text.lower()
    for name in names:
        parts = name.lower().split()
        if all(p in t for p in parts):
            return True
    return False

def _which_documents(q: str) -> list[str]:
    """
    Returns a unique, sorted list of filenames that mention the topic in the question.
    We scan ALL indexed chunks to avoid missing matches.
    """
    m = re.search(r"which\s+(?:documents|files)\s+(?:.*?\s)?(?:mention|talk about|cover)\s+(.+?)\??$", q, re.IGNORECASE)
    topic = (m.group(1).strip() if m else "").lower()
    if not topic:
        topic = " ".join([c.text for c in nlp(q).noun_chunks]).lower() or q.lower()

    tokens = [tok for tok in topic.split() if len(tok) > 2]
    hits = set()
    for fn, ch in documents:
        t = ch.lower()
        if all(tok in t for tok in tokens):
            hits.add(fn)
    return sorted(hits)



from dataclasses import dataclass
from sentence_transformers import CrossEncoder

@dataclass
class RetrievedItem:
    filename: str
    chunk: str
    score: float  

def _short_quote(text: str, max_len: int = 220) -> str:
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rsplit(" ", 1)[0] + "…"

def _confidence_label(scores: list[float]) -> str:
    if not scores:
        return "Low"
    avg = sum(scores) / len(scores)
    if avg >= 0.65:
        return "High"
    if avg >= 0.45:
        return "Medium"
    return "Low"


_RERANKER = None
def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        try:
            _RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print("⚠️ Reranker load failed:", e)
            _RERANKER = None
    return _RERANKER

def get_top_k_chunks(query: str, k: int = 8):
    """
    Retrieve top-k chunks grouped by file using FAISS (lower L2 = better).
    Returns: list of (filename, [best_chunks_for_that_file]) ordered by best score.
    """
    global faiss_index, documents, model, nlp
    if faiss_index is None or not documents:
        return []

   
    doc_q = nlp(query)
    important_parts = [nc.text for nc in doc_q.noun_chunks] + [ent.text for ent in doc_q.ents]
    cleaned_query = " ".join(important_parts) if important_parts else query

    qv = model.encode(cleaned_query).reshape(1, -1)
    D, I = faiss_index.search(np.array(qv).astype("float32"), k=min(k, len(documents)))

    # Collect (filename, chunk, distance)
    scored = []
    q_words = set(query.lower().split())
    for idx, dist in zip(I[0], D[0]):
        fn, ch = documents[idx]

       
        dist = float(dist)
        fn_l = fn.lower()
        if any(w in fn_l for w in q_words):
            dist *= 0.8
        if any(tag in fn_l for tag in ["support", "petition", "response", "denial", "approval", "rfe", "letter"]):
            dist *= 0.85

        scored.append((fn, ch, dist))

    
    by_file = {}
    for fn, ch, dist in scored:
        by_file.setdefault(fn, []).append((dist, ch))
    for fn in by_file:
        by_file[fn].sort(key=lambda x: x[0])
        by_file[fn] = [ch for _, ch in by_file[fn][:k]]

    # Order files by their best (lowest) distance
    best_per_file = {}
    for fn, _, dist in scored:
        best_per_file[fn] = min(dist, best_per_file.get(fn, dist))
    ordered = sorted(best_per_file.items(), key=lambda x: x[1])

    # Final: [(filename, [chunks])]
    return [(fn, by_file[fn]) for fn, _ in ordered]

def retrieve_top(query: str, top_k: int = 30, rerank_top: int = 8) -> list[RetrievedItem]:
    """
    1) Use FAISS to get top_k candidate chunks
    2) Re-rank them with a cross-encoder
    3) Return the best rerank_top items
    """
    by_file = get_top_k_chunks(query, k=top_k)  # [(filename, [chunks])]
    candidates = []
    for fn, chs in by_file:
        for ch in chs:
            if ch and len(ch.strip()) > 20:
                candidates.append((fn, ch))

    if not candidates:
        return []

    reranker = _get_reranker()
    if reranker is None:
        return [RetrievedItem(fn, ch, 0.5) for fn, ch in candidates[:rerank_top]]

    pairs = [(query, ch) for _, ch in candidates]
    try:
        scores = reranker.predict(pairs).tolist()
    except Exception as e:
        print("⚠️ Reranker predict failed:", e)
        return [RetrievedItem(fn, ch, 0.5) for fn, ch in candidates[:rerank_top]]

    scored = [RetrievedItem(fn, ch, float(s)) for (fn, ch), s in zip(candidates, scores)]
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:rerank_top]

def _uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_regulations(text: str) -> list[str]:
    """
    Extract regulation/clause refs like:
      8 CFR §214.2(h)(2)(i)(B)
      INA § 101(a)(15)(H)
      § 214.2(h)(13)(i)(A)
    Returns a de‑duplicated list (already normalized via normalize_citation).
    """
    pattern = r"""
        (?ix)
        (?:\b(?:8\s*CFR|8\s*\.?\s*C\.?F\.?R\.?|CFR)\b\s*(?:§|Section)?\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)
        |
        (?:\bINA\b\s*§\s*\d+(?:\([a-z0-9]+\))*)
        |
        (?:§\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)
    """
    matches = re.finditer(pattern, text, re.IGNORECASE | re.VERBOSE)
    out = []
    for m in matches:
        norm = normalize_citation(m.group(0))
        if norm and norm not in out:
            out.append(norm)
    return out

def _extract_denial_reason(text: str) -> Optional[str]:
    """
    Pulls a short, human-readable denial reason sentence.
    Looks for common H-1B denial phrasing and trims it.
    """
    t = " ".join(text.split())

    # Common denial cues
    candidates = re.findall(
        r"(?:(?:petition|case|request)\s+is\s+denied[^.]*\.)|"
        r"(?:failed\s+to\s+(?:establish|demonstrate)[^.]*\.)|"
        r"(?:has\s+not\s+established[^.]*\.)|"
        r"(?:does\s+not\s+qualify[^.]*\.)|"
        r"(?:position\s+does\s+not\s+qualify[^.]*\.)",
        t, re.IGNORECASE
    )

    if not candidates:
     
        m = re.search(r"(?:not\s+a\s+specialty\s+occupation[^.]*\.)", t, re.IGNORECASE)
        if m:
            candidates = [m.group(0)]

    if not candidates:
        return None

    # Clean & shorten
    s = candidates[0].strip()
    s = re.sub(r"\s+", " ", s)
    s = s[0].upper() + s[1:]
    return s



def _compose_answer(query: str, mode: str, items: list[RetrievedItem]) -> str:
    """
    Clean markdown:
      **Direct answer**
      <one concise line>

      **Key points**
      - bullet with short quote — filename
      - ...

      Confidence: <High/Medium/Low>
    """
    # ----- special mode: list files that mention a topic -----
    if mode == "which_docs":
        files = _which_documents(query)
        if not files:
            return "No documents in your upload mention that topic."
        direct = ", ".join(files[:3]) + (f" (+{len(files)-3} more)" if len(files) > 3 else "")
        bullets = [f"- **{fn}**" for fn in files[:8]]
        conf = "High" if len(files) else "Low"

        parts = [f"**Direct answer**\n{direct}\n"]
        if bullets:
            parts.append("**Key points**")
            parts.extend(bullets)
        parts.append(f"\nConfidence: {conf}")
        return "\n".join(parts)
        # --- Facts: petitioner / beneficiary / receipt number/date / decision date ---
    if mode.startswith("fact_"):
        kind = mode.replace("fact_", "")  
        res = _answer_fact(kind, items)
        if not res:
            return "⚠️ I couldn’t find that field in your documents."
        direct, bullets = res
        conf = _confidence_label([it.score for it in items])
        return "\n".join([
            "### Direct answer",
            direct,
            "",
            "### Key points",
            *bullets,
            "",
            f"**Confidence:** {conf}",
        ])


    if not items:
        return "No relevant information found in the uploaded documents."

    # join text for pattern extraction
    joined_text = " ".join([it.chunk for it in items])
    bullets: list[str] = []
    direct: str | None = None
    scores = [it.score for it in items]

    # ---------- helpers ----------
    def reg_sources(regs: list[str]) -> list[tuple[str, str]]:
        """map each regulation to the first item (filename) that contains it"""
        out = []
        for raw in regs:
            r_norm = normalize_citation(raw)
            found_fn = None
        
            needle_a = r_norm.lower()
            needle_b = needle_a.replace(" ", "")
            for it in items:
                hay = it.chunk.lower()
                if needle_a in hay or needle_b in hay.replace(" ", ""):
                    found_fn = it.filename
                    break
            out.append((r_norm, found_fn or "—"))
        return out

  

    if is_regulation_query(query) or mode == "regulation":
        regs = _uniq(extract_regulations(joined_text))
        if not regs:
            return "No regulation or clause found in the uploaded documents."
        regs = regs[:4]
        direct = "Regulations cited: " + ", ".join([f"**{normalize_citation(r)}**" for r in regs])
        for r, fn in reg_sources(regs):
            bullets.append(f"- {r} — found in **{fn}**")
        conf = "High"  


    elif mode == "rfe":
 
        concern = None
        chosen: RetrievedItem | None = None
        for it in items:
            concern = _extract_rfe_concern(it.chunk)
            if concern:
                chosen = it
                break
        if not concern:
            return "No specific RFE concern found in the uploaded documents."
        direct = f"RFE concern: **{concern}**."
        bullets.append(f"- {_short_quote(chosen.chunk)} — **{chosen.filename}**")

    elif mode == "denial":
     
        reasons: list[tuple[str, str]] = []
        for it in items:
            r = _extract_denial_reason(it.chunk)
            if r:
                reasons.append((r, it.filename))
 
        seen = set()
        uniq = []
        for r, fn in reasons:
            key = r.lower()
            if key not in seen:
                seen.add(key)
                uniq.append((r, fn))
        if not uniq:
            return "No denial reasons detected in the uploaded documents."
        direct = uniq[0][0]
        for r, fn in uniq[:3]:
            bullets.append(f"- {r.rstrip('.')} — **{fn}**")

    else:
       
        for it in items[:3]:
            try:
                qa = qa_pipeline({"question": query, "context": it.chunk})
                ans = qa.get("answer", "").strip()
                if len(ans.split()) >= 3:
                    direct = ans[0].upper() + ans[1:].rstrip(".") + "."
                    break
            except Exception:
                pass
        if not direct:
            direct = "Based on the uploaded documents, there isn’t a clear one‑line answer."
        for it in items[:3]:
            bullets.append(f"- {_short_quote(it.chunk)} — **{it.filename}**")


    
    if 'conf' not in locals():
        conf = _confidence_label(scores)


    parts = [f"**Direct answer**\n{direct}\n"]
    if bullets:
        parts.append("**Key points**")
        parts.extend(bullets)
    parts.append(f"\nConfidence: {conf}")
    return "\n".join(parts)



def _detect_query_mode(q: str) -> str:
    ql = q.lower()

    if re.search(r"\bwhen\s+(?:is|was)\s+(?:the\s+)?h-?1b(?:\s+petition)?\s+filed\b", ql):
        return "fact_receipt_date"

    if re.search(r"which\s+(documents|files)\b", ql) and re.search(r"\b(mention|talk about|cover)\b", ql):
        return "which_docs"
    if re.search(r"\bwho\b.*\bpetitioner\b", ql):
        return "fact_petitioner"
    if re.search(r"\bwho\b.*\bbeneficiary\b", ql):
        return "fact_beneficiary"
    if "receipt number" in ql:
        return "fact_receipt_number"
    if "receipt date" in ql:
        return "fact_receipt_date"
    if "decision date" in ql:
        return "fact_decision_date"
    if "regulation" in ql or "clause" in ql or "§" in ql or "cfr" in ql or "ina" in ql:
        return "regulation"
    if "rfe" in ql or "request for evidence" in ql or "concern" in ql:
        return "rfe"
    if "deny" in ql or "denied" in ql or ("why" in ql and "petition" in ql):
        return "denial"
    if "approval" in ql or "approved" in ql or "i-797" in ql:
        return "approval"
    return "general"




def _which_documents(q: str) -> list[str]:
    """
    Returns a unique, sorted list of filenames that mention the topic in the question.
    We scan ALL indexed chunks to avoid missing matches.
    """
   
    m = re.search(r"which\s+(?:documents|files)\s+(?:.*?\s)?(?:mention|talk about|cover)\s+(.+?)\??$", q, re.IGNORECASE)
    topic = (m.group(1).strip() if m else "").lower()
    if not topic:
    
        topic = " ".join([c.text for c in nlp(q).noun_chunks]) or q

    hits = set()
    for fn, ch in documents:
        t = ch.lower()
        if all(tok in t for tok in topic.split() if len(tok) > 2):
            hits.add(fn)
    return sorted(hits)


_MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
FACT_PATTERNS = {

    "petitioner": re.compile(
        r"\bPetitioner\s*[:\-]\s*([A-Za-z0-9&\.,\- ]{2,100}?)\s*(?=\b(Beneficiary|Receipt|Decision|Date|Case|Summary)\b|$)",
        re.IGNORECASE
    ),
    "beneficiary": re.compile(
        r"\bBeneficiary\s*[:\-]\s*([A-Za-z][A-Za-z0-9\.\- ]{1,100}?)\s*(?=\b(Receipt|Decision|Date|Case|Summary|Petitioner)\b|$)",
        re.IGNORECASE
    ),
    "receipt_number": re.compile(
        r"\bReceipt(?:\s*(?:No\.?|Number))?\s*[:\-]\s*([A-Z]{3}\d{8,})",
        re.IGNORECASE
    ),
    "receipt_date": re.compile(
        rf"\b(?:Receipt\s*Date|Date)\s*[:\-]\s*{_MONTHS}\s+\d{{1,2}},\s+\d{{4}}",
        re.IGNORECASE
    ),
    "decision_date": re.compile(
        rf"\bDecision\s*Date\s*[:\-]\s*{_MONTHS}\s+\d{{1,2}},\s+\d{{4}}",
        re.IGNORECASE
    ),
}


def _extract_fact_from_text(kind: str, text: str) -> Optional[str]:
    m = FACT_PATTERNS[kind].search(text)
    if not m:
        return None
    out = m.group(0)
 
    if kind in ("receipt_date", "decision_date"):
        m2 = re.search(rf"{_MONTHS}\s+\d{{1,2}},\s+\d{{4}}", out, re.IGNORECASE)
        return (m2.group(0).strip() if m2 else None)
    elif kind in ("petitioner", "beneficiary", "receipt_number"):
        m2 = re.search(r"[:\-]\s*(.+)$", out)
        return (m2.group(1).strip() if m2 else None)
    return None

def _answer_fact(kind: str, items: list[RetrievedItem]) -> Optional[tuple[str, list[str]]]:
    """
    Returns (direct, bullets) or None if not found.
    Tries top reranked chunks, then falls back to the CASES index.
    """
    values = []
    bullets = []

    # rank chunks
    for it in items:
        val = _extract_fact_from_text(kind, it.chunk)
        if val:
            values.append((val, it.filename, it.chunk))

 
    if not values:
        for fn, meta in CASES.items():
            val = meta.get(kind)
            if val:
                quote = f"{kind.replace('_',' ').title()}: {val}"
                values.append((val, fn, quote))

    if not values:
        return None

    # pick first 
    top = values[0][0]
    direct = {
        "petitioner": f"The petitioner is **{top}**.\n",
        "beneficiary": f"The beneficiary is **{top}**.\n",
        "receipt_number": f"Receipt number: **{top}**.\n",
        "receipt_date": f"Receipt date: **{top}**.\n",
        "decision_date": f"Decision date: **{top}**.\n",
    }[kind]

    # up to 3 supporting bullets
    for val, fn, ch in values[:3]:
        bullets.append(f"- {val} — **{fn}**\n  > “{_short_quote(ch)}”")
    return direct, bullets


def ask_question(query):
    print("USER QUESTION:", query)
    ql = query.lower()


    generic_proc = re.search(r"\bwhen\s+(?:is|was)\s+h-?1b\b", query, re.IGNORECASE)
    scoped = re.search(r"\b(this case|this petition|this file|in these documents)\b", query, re.IGNORECASE)
    if generic_proc and not scoped:
        return (
            "**Direct answer**\n"
            "I can only answer from your uploaded case documents. If you’re asking about the general USCIS H‑1B filing window, that’s outside these files. "
            "Try: “When was **this petition** filed?” or “What is the **receipt date**?”\n\n"
            "Confidence: Low"
        )

    reg_mode = is_regulation_query(query)
    mode = _detect_query_mode(query)
    if reg_mode:
        mode = "regulation"

    if not embeddings:
        return "❌ No documents indexed. Please upload and refresh first."

    # Step 1: 
    items = retrieve_top(query, top_k=30, rerank_top=8)

    # Step 2: (keep only items mentioning the entity)
    ents = [ent.text.strip() for ent in nlp(query).ents if ent.label_ in ("PERSON", "ORG")]
    ent_norm = [e for e in ents if e]
    if ent_norm:
        filtered = [it for it in items if _mentions_any_entity(it.chunk, ent_norm)]
        if filtered:
            items = filtered

    # --- Off-topic guard ---
    structured_modes = {"regulation", "which_docs", "rfe", "denial"}
    if mode.startswith("fact_"):
        structured_modes.add(mode)
    if (mode not in structured_modes) and (not items or _best_score(items) < OFFTOPIC_MIN_SCORE):
        return (
            "**Direct answer**\n"
            "Your question doesn’t appear to match the uploaded immigration documents. "
            "Try asking about RFEs, denials, regulations, receipt details, petitioners/beneficiaries, etc.\n\n"
            "Confidence: Low"
        )

    # Step 3:
    return _compose_answer(query, mode, items)



# Only run these on startup
load_documents()
print("SAMPLE DOCUMENT CHUNKS:", documents[:2])
print("SAMPLE EMBEDDINGS:", embeddings[:2])
faiss_index = build_faiss_index()
