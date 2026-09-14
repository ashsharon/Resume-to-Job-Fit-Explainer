# Resume-to-Job Fit Explainer

A tool that goes beyond a single "match score" between a resume and a job posting.
For any resume + job description pair, it produces:

1. **Requirement-by-requirement coverage** — extracts every requirement from the job
   description and checks whether the resume covers it (via semantic embedding similarity,
   not just keyword matching).
2. **A structured gap analysis** — for each missing/weak requirement, explains whether the
   candidate has *transferable* experience or a genuine gap.
3. **Rewritten resume bullet points** — reframes existing bullets (without inventing new
   experience) to better surface relevance to this specific job.

This is the unique feature relative to typical resume-matching tools: the output is
**actionable** (what to fix, how to fix it) rather than a single opaque percentage.

## Architecture

```
Resume file (PDF/TXT) + Job Description (text)
        │
        ▼
resume_parser.py   ──► extracts plain text from PDF/TXT
        │
        ▼
matcher.py          ──► (a) LLM extracts structured requirement list from JD
        │                (b) sentence embeddings score each requirement against resume lines
        ▼
analyzer.py          ──► LLM generates gap analysis + rewritten bullet points,
        │                 grounded in the requirement coverage data above
        ▼
pipeline.py           ──► orchestrates all steps into one fit report
        │
        ▼
app.py                 ──► Streamlit UI
```

## Project structure

```
resume-fit-explainer/
├── app.py                     # Streamlit frontend
├── pipeline.py                 # Orchestrates parsing + matching + analysis
├── resume_parser.py             # PDF/TXT text extraction
├── matcher.py                    # Requirement extraction + embedding-based scoring
├── analyzer.py                    # LLM gap analysis + bullet rewriting
├── requirements.txt
├── .env.example
└── data/
    ├── sample_job_description.txt
    └── sample_resume.txt
```

## Setup (VS Code / local)

1. **Open this folder in VS Code.**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Anthropic API key:**
   ```bash
   export ANTHROPIC_API_KEY=your-key-here     # macOS/Linux
   set ANTHROPIC_API_KEY=your-key-here        # Windows (cmd)
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`. Upload a resume PDF (or use the sample .txt in
   `data/`) and paste in a job description.

6. **Or run the full pipeline from the command line:**
   ```bash
   python pipeline.py
   ```
   This uses the bundled sample resume + job description and prints a JSON report.

## Testing individual components

```bash
python resume_parser.py    # test text extraction on the sample resume
python matcher.py            # test requirement extraction + scoring
python analyzer.py            # test full gap analysis + bullet rewriting
```

## Notes on the sample data

The bundled sample resume (`data/sample_resume.txt`) is deliberately a **data analyst**
resume being matched against a **machine learning engineer** job description — this is
realistic and shows partial fit with genuine gaps (no PyTorch/TensorFlow, no cloud
deployment, no MLOps) alongside real transferable skills (Python, SQL, scikit-learn),
which makes for a much more interesting demo/report than a perfect match.

## Extending it (ideas for your report / future work section)

- Support DOCX resumes (add `python-docx` parsing alongside the PDF path).
- Add a batch mode to score one resume against many job postings, or many resumes against
  one posting (useful for a recruiter-facing variant).
- Fine-tune the similarity threshold in `matcher.py` per requirement category, or
  weight "required" vs "preferred" differently in the fit score formula.
- Add an evaluation set of resume/JD pairs with human-labeled fit scores to report
  correlation between the tool's fit score and human judgment.
- Cache extracted requirements per job description so repeated resume checks against the
  same posting don't re-call the LLM for extraction every time.

## Notes

- The first run downloads the `all-MiniLM-L6-v2` embedding model (~80MB).
- Scanned/image-only PDFs are not supported (no OCR) — use a text-based PDF or the
  `.txt` fallback.
