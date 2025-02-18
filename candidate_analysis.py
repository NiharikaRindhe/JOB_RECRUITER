import json
import streamlit as st
import google.generativeai as genai

# 🔑 Hardcoded Gemini API Key
GEMINI_API_KEY = "AIzaSyCPg-kwf9cAAjHkMCDVYY_t9yLNgC_StvM"  # Replace with your actual API key

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Define Model Configuration
generation_config = {
    "temperature": 0.7,  # Lower randomness for more relevant results
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,  # Reduce response size to prevent API failures
    "response_mime_type": "text/plain",
}

# Initialize Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",  # Use a stable version instead of preview models
    generation_config=generation_config,
)

# Function to analyze candidate suitability using Gemini API
def analyze_candidate(jd, resume):
    prompt = f"""
    You are an AI assistant that helps recruiters analyze job candidates.
    Compare the Job Description (JD) and the Candidate Resume.

    **Important Instructions:**
    - Check the required skills from the JD in all resume sections: **skills, experience, and projects**.
    - If a skill appears anywhere in the resume, classify it as **Matching**.
    - If a skill is completely missing from the resume, classify it as **Non-Matching**.

    **Job Description (JD):**
    {json.dumps(jd, indent=2)}

    **Candidate Resume:**
    {json.dumps(resume, indent=2)}

    **Output format:**

    **Matching Skills:**
    - List only the skills that are found in the resume (in skills, experience, or projects).

    **Non-Matching Skills:**
    - List only the skills from the JD that are NOT found anywhere in the resume.

    **Final Fit Summary:**
    - Provide a short paragraph about how well the candidate fits the job.
    - Mention whether an interview is recommended based on gaps and strengths.
    """

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        return response.text  # Return the structured response directly

    except Exception as e:
        st.error(f"⚠️ API Error: {str(e)}")
        return None

# Streamlit UI
st.title("🔍 Job Candidate Analysis using Gemini AI")
st.write("Enter a **Job Description (JD) and Candidate Resume** in JSON format to analyze their fit.")

# Text Area Inputs for JD and Resume
jd_input = st.text_area("📜 Enter Job Description (JSON)", height=200, placeholder='{"title": "Software Engineer", "required_skills": ["Python", "Django"], "required_experience": ["Backend Developer"], "required_technologies": ["AWS"]}')
resume_input = st.text_area("📝 Enter Candidate Resume (JSON)", height=200, placeholder='{"name": "John Doe", "skills": ["Python", "Flask"], "experience": ["Backend Developer"], "projects": ["Built an API using GraphQL"], "technologies": ["AWS"]}')

# Process the Inputs
if st.button("🚀 Analyze Candidate"):
    try:
        jd_data = json.loads(jd_input)
        resume_data = json.loads(resume_input)

        with st.spinner("Analyzing..."):
            analysis_result = analyze_candidate(jd_data, resume_data)

            if analysis_result:
                st.subheader("📄 Candidate Analysis Summary")
                st.markdown(analysis_result)  # Display structured markdown response
    except json.JSONDecodeError:
        st.error("⚠️ Invalid JSON format! Please check your input.")
 
