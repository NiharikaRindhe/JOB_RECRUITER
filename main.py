import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import google.generativeai as genai

# API Key Setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyCPg-kwf9cAAjHkMCDVYY_t9yLNgC_StvM"))

# Gemini model config
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

# Models
model_pro = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

model_flash = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

# FastAPI app
app = FastAPI()

class AnalyzeRequest(BaseModel):
    jd: Dict[str, Any]
    resume: Dict[str, Any]

class QuestionRequest(BaseModel):
    resume: Dict[str, Any]
    jd: Optional[Dict[str, Any]] = None

def analyze_candidate(jd, resume):
    prompt = f"""
You are an AI assistant that helps recruiters analyze how well a candidate fits a job based on the Job Description (JD) and Resume.

**Your Tasks:**
1. Identify the required skills from the JD (from the `skills` key only).
2. Look for those skills **only in these resume sections**:
   - `skillsDetails.skills`
   - `professionalDetails[*].skills`
   - `projectDetails[*].skills`
3. If a skill appears in any of the above, add it under **Matching Skills**.
4. If a skill does NOT appear in any of the above, add it under **Non-Matching Skills**.

**Important Notes:**
- Do NOT say “Recruitment (Not Found)” — just list matching skills under one heading, and missing ones under the other.
- If there are **no matching skills**, write **"No Matching Skills"** under that section.
- Your final summary must be clear, structured, and in **bullet points**, not paragraph form.
- Conclude with whether an **interview is recommended** or not.

---

**Job Description:**
{json.dumps(jd, indent=2)}

**Candidate Resume:**
{json.dumps(resume, indent=2)}

---

**Output Format:**

** Matching Skills:**
- Skill 1
- Skill 2
(If none, write: "No Matching Skills")

** Non-Matching Skills:**
- Skill A
- Skill B

**📋 Final Fit Summary:**
- The candidate has strong experience in [insert main areas of resume].
- However, they lack required HR skills like X, Y, Z.
- There is no relevant experience in HR domains.
- **Interview is not recommended** for this role.
"""
    try:
        chat = model_pro.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

def generate_interviewer_questions(resume, jd=None):
    jd_section = f"\nJob Description:\n{json.dumps(jd, indent=2)}" if jd else ""
    source_instruction = (
        "Blend Resume and JD to infer the domain and generate interviewer questions accordingly."
        if jd else
        "Use only the Resume to infer the domain and generate interviewer questions."
    )

    prompt = f"""
You are a senior professional preparing to interview a candidate.

📄 Based on the following resume{' and job description' if jd else ''}, infer the candidate’s domain (technical or non-technical).
🌟 Your task is to generate 20 focused interview questions that are most relevant to the candidate’s background:
- If technical domain (e.g., Software, Engineering, Data, etc.): generate **only technical** questions.
- If non-technical domain (e.g., Marketing, HR, Education, etc.): generate **only domain-relevant** questions.
- ❌ Do NOT mix types or include general behavioral/non-technical questions for technical candidates.
- ❌ Do NOT add headings like "Technical" or "Non-Technical".
- ❌ Do NOT include answers.

✅ Format each like:
[number]. Q: [question]

{source_instruction}

{jd_section}

Resume:
{json.dumps(resume, indent=2)}

Now generate only the 20 questions accordingly.
"""

    try:
        chat = model_flash.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

@app.post("/CANDIDATE_ANALYSIS")
async def analyser(request: AnalyzeRequest):
    result = analyze_candidate(request.jd, request.resume)
    return {"analysis": result}

@app.post("/GENERATE_QUESTIONS")
async def generate_questions(request: QuestionRequest):
    result = generate_interviewer_questions(request.resume, request.jd)
    return {"questions": result}
