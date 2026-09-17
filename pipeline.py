"""
pipeline.py
-----------
Top-level orchestration: resume file + job description -> full fit report.
"""

import os
from resume_parser import parse_resume
from matcher import ResumeJobMatcher
from analyzer import FitAnalyzer


class ResumeFitPipeline:
    def __init__(self, api_key: str | None = None):
        self.matcher = ResumeJobMatcher(api_key=api_key)
        self.analyzer = FitAnalyzer(api_key=api_key)

    def run(self, resume_path: str, job_description: str) -> dict:
        resume_text = parse_resume(resume_path)

        requirements = self.matcher.extract_requirements(job_description)
        scored = self.matcher.score_requirements(requirements, resume_text)
        analysis = self.analyzer.analyze(job_description, resume_text, scored)

        required_total = len(scored["required"])
        required_covered = sum(1 for r in scored["required"] if r["covered"])
        fit_score = round(100 * required_covered / required_total) if required_total else 0

        return {
            "fit_score_percent": fit_score,
            "resume_text": resume_text,
            "requirement_coverage": scored,
            "fit_summary": analysis["fit_summary"],
            "gap_analysis": analysis["gap_analysis"],
            "rewritten_bullets": analysis["rewritten_bullets"]
        }


if __name__ == "__main__":
    import json

    here = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(here, "data", "sample_resume.txt")
    with open(os.path.join(here, "data", "sample_job_description.txt")) as f:
        jd = f.read()

    pipeline = ResumeFitPipeline()
    result = pipeline.run(resume_path, jd)
    print(json.dumps(result, indent=2))
