Allergy Opinion Generator

The Allergy Opinion Generator is a Streamlit-based application that analyzes allergy-related .docx test reports and generates clear, doctor-style opinions. It combines LLMs (OpenAI), LangChain prompt orchestration, and FAISS-powered retrieval to answer questions about a report and summarize key findings with citations to the source text.


---

Features

Upload & parse allergy reports (.docx)

Generate expert-style opinions with structured templates

Interactive Q&A grounded in the uploaded report

Retrieval-augmented generation (RAG) via FAISS vector search

Export generated opinions to .docx



---

Tech Stack

Python 3.9+

Streamlit (UI)

LangChain (prompting & pipelines)

OpenAI API (LLM)

FAISS (vector store / semantic search)

python-docx (document I/O)

Jupyter Notebook (experiments & demos)



---

Quickstart

1) Clone & set up

git clone https://github.com/abhishek-a-qbit/Allergy-Opinion-Generator.git
cd Allergy-Opinion-Generator

# (recommended) create a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\\Scripts\\Activate.ps1

# install dependencies
pip install -r requirements.txt

2) Configure API key

# macOS/Linux
export OPENAI_API_KEY="your_api_key_here"

# Windows (PowerShell)
setx OPENAI_API_KEY "your_api_key_here"

3) Run the app

streamlit run main.py

Open the local URL Streamlit prints (usually http://localhost:8501).

Upload an allergy test report (.docx).

Review the generated opinion summary and ask follow-up questions.

(Optional) Export the opinion to .docx.



---

Typical Workflow

1. Ingest: Upload a patient allergy report (.docx).


2. Chunk & Embed: The text is chunked and embedded, then stored in FAISS.


3. RAG: Questions/opinion prompts fetch the most relevant chunks.


4. LLM Generation: LangChain templates + OpenAI produce an expert-style summary/opinion.


5. Output: Display in the UI and (optionally) save to .docx.




---

Repository Structure
## Repository Structure

```plaintext
Allergy-Opinion-Generator/
├── .devcontainer/           # Dev container config
├── main.py                  # Streamlit application entry point
├── model.py                 # Core retrieval & LLM orchestration
├── requirements.txt         # Project dependencies
├── test.ipynb               # Notebook for experiments/demos
├── Indoor Allergen.docx     # Sample domain document
├── Pollens.docx             # Sample domain document
├── MAMTHA A9151.docx        # Sample patient report
├── PARVESH A2112.docx       # Sample patient report
├── temp_response.docx       # Example generated opinion
└── temp_test_report.docx    # Example processed report

```
---

Notebooks

test.ipynb demonstrates the pipeline on example inputs, useful for quick iteration and validation. (Languages reported by GitHub: ~69% Notebook, ~31% Python.)



---

Development Notes

Dev Container: .devcontainer/ is included for a reproducible setup (VS Code + Docker).

Prefer small, well-named functions in model.py for retrieval and prompt steps.

Keep prompts/templates version-controlled to track changes in output style.



---

Security & Data

Do not upload real patient data unless you have explicit consent.

Redact identifiers where possible before ingestion.

Be mindful of OpenAI API usage policies and PHI handling norms.



---

Troubleshooting

Streamlit doesn’t start: Confirm environment is activated and pip install -r requirements.txt completed without errors.

API errors: Ensure OPENAI_API_KEY is set in the same shell where you launch Streamlit.

Long documents: Increase chunk size/overlap thoughtfully; ensure FAISS index fits memory.



---

License

This project is released under CC0-1.0 (public domain dedication). You may use, modify, and distribute without restriction.


---

Status

No releases published at the time of writing; use the main branch.


---

Author

Abhishek A. — MSc Mathematics | AI Developer | Generative AI

