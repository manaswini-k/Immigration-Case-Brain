Immigration Case Brain
Immigration Case Brain is a simple Streamlit app I built to help quickly search through immigration-related case files like H-1B RFEs, denial notices, support letters, and petitions.
You can upload your documents, ask a question in plain English, and get an answer that’s pulled directly from the relevant parts of the file all without relying on paid APIs.

->Sample Data
Inside the data/ folder, you’ll find a few redacted sample documents to try it out:

H1B Approval Notice Memo.pdf

H1B Support Anjali Verma.pdf

H1B_Denial_Inhouse_Employment.pdf

H1B_RFE_Education_Mismatch.pdf

These files are just for testing the app.
Don’t upload real or private case documents to a public repo.

->What It Can Do
Upload multiple PDF/DOCX files at once

Ask a question and get an answer specific to the document it came from

See citations and short snippets for context

Works completely offline after setup

Switch between light and dark themes via .streamlit/config.toml

->Example Questions
Once your files are uploaded, you could ask things like:

What was the main reason for the RFE?

Why was the petition denied?

What regulation was cited in the denial?

Who is the petitioner?

When was the H-1B filed?

Who is the beneficiary?

The app runs a semantic search across your documents, finds the best matches, and returns clear answers with references.

->Tech Stack
Frontend/UI: Streamlit

NLP: spaCy

Semantic Search: Sentence Transformers

ML Backend: PyTorch

Models Used: all-MiniLM-L6-v2 (Sentence Transformers), en_core_web_sm (spaCy)

->How It Works
Upload & Parse – Reads your PDFs/DOCX and breaks them into text chunks.

Vectorize – Converts each chunk into an embedding using Sentence Transformers.

Search – Embeds your question and finds the most similar chunks with cosine similarity.

Extract Answer – Pulls the top matches and formats them into a clean answer (with optional previews).

Show Results – Displays the answer, source file, and snippets in the Streamlit UI.

->Run It Locally
# 1. Create and activate the conda environment (name it spacy for consistency)
conda create -n spacy python=3.10 -y
conda activate spacy

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install faiss separately (since it’s not always installed cleanly from requirements.txt)
pip install faiss-cpu

# 4. Download the spaCy model
python -m spacy download en_core_web_sm

# 5. (Optional) Verify installation
python -c "import faiss, spacy; print('FAISS version:', faiss.__version__); print('spaCy version:', spacy.__version__)"

# 6. Start the Streamlit app
streamlit run app/main.py

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Start the app
streamlit run app/main.py
