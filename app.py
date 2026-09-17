"""
app.py
------
Streamlit frontend for the Resume-to-Job Fit Explainer.

Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from pipeline import ResumeFitPipeline

st.set_page_config(page_title="Resume-to-Job Fit Explainer", page_icon="📄", layout="centered")

st.title("📄 Resume-to-Job Fit Explainer")
st.caption("Upload a resume and paste a job description to get a requirement-by-requirement "
           "gap analysis and rewritten bullet points - not just a match score.")

with st.sidebar:
    st.header("About")
    st.write(
        "This tool extracts structured requirements from the job description, semantically "
        "matches each one against your resume, then generates an actionable gap analysis "
        "and rewrites relevant bullet points - without inventing experience you don't have."
    )
    st.divider()
    st.write("Sample files are in the `data/` folder if you want to try it without your own resume.")


@st.cache_resource(show_spinner="Loading models...")
def load_pipeline():
    return ResumeFitPipeline()


pipeline = load_pipeline()

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
job_description = st.text_area("Paste the job description:", height=250)

if st.button("Analyze Fit", type="primary"):
    if not uploaded_file or not job_description.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Analyzing fit..."):
                result = pipeline.run(tmp_path, job_description)
        finally:
            os.unlink(tmp_path)

        st.markdown(f"### Fit Score: {result['fit_score_percent']}% of required qualifications covered")
        st.progress(result["fit_score_percent"] / 100)
        st.write(result["fit_summary"])

        st.divider()
        st.subheader("Requirement Coverage")
        for category, label in [("required", "Required"), ("preferred", "Preferred")]:
            st.write(f"**{label}**")
            for item in result["requirement_coverage"][category]:
                icon = "✅" if item["covered"] else "❌"
                st.write(f"{icon} {item['requirement']}  _(score: {item['score']})_")

        st.divider()
        st.subheader("Gap Analysis")
        for gap in result["gap_analysis"]:
            badge = "🟡 Transferable" if gap["status"] == "transferable" else "🔴 Gap"
            st.write(f"**{gap['requirement']}** — {badge}")
            st.write(gap["note"])

        st.divider()
        st.subheader("Suggested Bullet Point Rewrites")
        for b in result["rewritten_bullets"]:
            st.write(f"**Original:** {b['original']}")
            st.write(f"**Rewritten:** {b['rewritten']}")
            st.caption(b["reason"])
            st.write("")
