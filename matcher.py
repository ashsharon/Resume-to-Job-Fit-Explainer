"""
matcher.py
----------
Two jobs:
1. Extract discrete requirements from a raw job description using an LLM
   (turns unstructured JD text into a clean list of skill/qualification strings).
2. For each requirement, semantically compare it against the resume text using
   sentence embeddings to compute a per-requirement coverage score.

This gives a requirement-by-requirement breakdown instead of one opaque
"match score", which is what makes the gap analysis in analyzer.py possible.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic


REQUIREMENT_EXTRACTION_PROMPT = """You are given a job description. Extract every distinct \
requirement, skill, or qualification mentioned (required or preferred) as a clean, short list.

Rules:
- One requirement per line item, phrased concisely (max ~12 words)
- Split compound requirements into separate items (e.g. "Python and SQL" -> two items)
- Include both required AND preferred qualifications, but mark which is which
- Do not invent requirements that aren't in the text

Respond ONLY with valid JSON, no markdown fences, in this exact shape:
{
  "required": ["requirement 1", "requirement 2", ...],
  "preferred": ["requirement 1", "requirement 2", ...]
}
"""


class ResumeJobMatcher:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.embedder = SentenceTransformer(embedding_model)

    def extract_requirements(self, job_description: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=REQUIREMENT_EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": job_description}]
        )
        raw = response.content[0].text.strip().replace("```json", "").replace("```", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"required": [], "preferred": []}

    def score_requirements(self, requirements: dict, resume_text: str, threshold: float = 0.42) -> dict:
        """
        For each requirement, chunk the resume into lines/bullets and find the
        single best-matching resume line via cosine similarity. A requirement is
        considered 'covered' if its best match exceeds the threshold.
        """
        resume_lines = [line.strip("-• \t") for line in resume_text.split("\n") if line.strip()]
        resume_lines = [l for l in resume_lines if len(l) > 3]

        if not resume_lines:
            resume_lines = [resume_text]

        resume_embeddings = self.embedder.encode(resume_lines, convert_to_numpy=True, normalize_embeddings=True)

        results = {"required": [], "preferred": []}

        for category in ["required", "preferred"]:
            for req in requirements.get(category, []):
                req_embedding = self.embedder.encode([req], convert_to_numpy=True, normalize_embeddings=True)[0]
                scores = resume_embeddings @ req_embedding
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])

                results[category].append({
                    "requirement": req,
                    "covered": best_score >= threshold,
                    "score": round(best_score, 3),
                    "best_matching_line": resume_lines[best_idx] if best_score >= threshold else None
                })

        return results


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "sample_job_description.txt")) as f:
        jd = f.read()
    with open(os.path.join(here, "data", "sample_resume.txt")) as f:
        resume = f.read()

    matcher = ResumeJobMatcher()
    reqs = matcher.extract_requirements(jd)
    print(json.dumps(reqs, indent=2))

    scored = matcher.score_requirements(reqs, resume)
    print(json.dumps(scored, indent=2))
