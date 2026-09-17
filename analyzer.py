"""
analyzer.py
-----------
Takes the scored requirement coverage (from matcher.py) plus the original resume
and job description, and asks the LLM to produce:
1. A concise gap analysis (what's missing, what's transferable)
2. Rewritten resume bullet points that better surface relevant experience

This is the "actionable output" feature that distinguishes this project from a
plain match-score tool: the output is something the candidate can directly use.
"""

import json
import anthropic


ANALYSIS_PROMPT = """You are a career coach helping a candidate improve their resume for a \
specific job. You are given:
- The job description
- The candidate's resume
- A structured breakdown of which requirements are covered vs. missing (from semantic matching)

Your job:
1. Write a short overall fit summary (2-3 sentences).
2. For each MISSING or WEAKLY-covered required qualification, explain briefly whether the \
candidate likely has transferable experience for it (based on the resume) or if it's a genuine gap.
3. Rewrite 2-4 of the candidate's EXISTING resume bullet points to better highlight relevance \
to this specific job - do not invent experience they don't have, only reframe/reword what's \
already there to use more relevant terminology and emphasize impact.

Respond ONLY with valid JSON, no markdown fences, in this exact shape:
{
  "fit_summary": "string",
  "gap_analysis": [
    {"requirement": "string", "status": "gap" | "transferable", "note": "string"}
  ],
  "rewritten_bullets": [
    {"original": "string", "rewritten": "string", "reason": "string"}
  ]
}
"""


class FitAnalyzer:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze(self, job_description: str, resume_text: str, scored_requirements: dict) -> dict:
        user_message = (
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"RESUME:\n{resume_text}\n\n"
            f"REQUIREMENT COVERAGE (from semantic matching):\n"
            f"{json.dumps(scored_requirements, indent=2)}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=ANALYSIS_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        raw = response.content[0].text.strip().replace("```json", "").replace("```", "")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "fit_summary": f"Could not parse model output. Raw: {raw}",
                "gap_analysis": [],
                "rewritten_bullets": []
            }


if __name__ == "__main__":
    import os
    from matcher import ResumeJobMatcher

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "sample_job_description.txt")) as f:
        jd = f.read()
    with open(os.path.join(here, "data", "sample_resume.txt")) as f:
        resume = f.read()

    matcher = ResumeJobMatcher()
    reqs = matcher.extract_requirements(jd)
    scored = matcher.score_requirements(reqs, resume)

    analyzer = FitAnalyzer()
    result = analyzer.analyze(jd, resume, scored)
    print(json.dumps(result, indent=2))
