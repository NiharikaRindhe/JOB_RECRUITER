import os
import json
import re
import io
import pandas as pd
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBrc0vyseVt5Ed3-BK7jobOAN4I12R1E8Q")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# ---------------------------- MODELS ----------------------------
class AnalyzeRequest(BaseModel):
    jd: Dict[str, Any]
    resume: Dict[str, Any]

class QuestionRequest(BaseModel):
    resume: Dict[str, Any]
    jd: Optional[Dict[str, Any]] = None

class JobDescription(BaseModel):
    job_title: str
    skills_required: list
    experience: str

class JobInput(BaseModel):
    job_title: str
    location: List[str]
    job_description: str
    skills: List[str]
    experience: str

# ------------------------ CANDIDATE ANALYSIS ------------------------
@app.get("/")
async def root():
    return {"message": "FastAPI is alive!"}

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
- Your final summary must be clear,in detailed, structured, and in **bullet points**, not paragraph form.
- Conclude with whether an **interview is recommended** or not explain that in detail.
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
- **Interview is not recommended** for this roleand also write why .
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

@app.post("/CANDIDATE_ANALYSIS")
async def analyser(request: AnalyzeRequest):
    result = analyze_candidate(request.jd, request.resume)
    return {"analysis": result}

# ------------------------ QUESTION GENERATOR ------------------------
def generate_interviewer_questions(resume, jd=None):
    jd_section = f"\nJob Description:\n{json.dumps(jd, indent=2)}" if jd else ""
    source_instruction = (
        "Blend Resume and JD to infer the domain and generate interviewer questions accordingly."
        if jd else
        "Use only the Resume to infer the domain and generate interviewer questions."
    )

    prompt = f"""
You are a senior interviewer preparing a mock interview for a candidate.

🌟 Based on the resume{' and job description' if jd else ''}, infer the most suitable role or domain.
Then generate **20 interview questions with full, descriptive answers**:
- For technical roles: prioritize domain-specific 70 % technical questions 30% Nontechnical questions.
- For non-technical roles: include a mix of domain-relevant and behavioral questions.

✅ Format:
- Use clear section headers as needed  (Technical, Non Technical) for technical roles only
- Number each Q&A from **1 to 20** 
- For each:
  [number]. Q: [question]
     A: [answer]

- Ensure Q and A are on separate lines
- Infer answers professionally even if information is limited
- Write the answers in short. 
{source_instruction}

{jd_section}

Resume:
{json.dumps(resume, indent=2)}

Now generate only the 20 questions accordingly.
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

@app.post("/GENERATE_QUESTIONS")
async def generate_questions(request: QuestionRequest):
    result = generate_interviewer_questions(request.resume, request.jd)
    return {"questions": result}

# ------------------------ MCQ GENERATOR ------------------------
def extract_details_from_jd(jd_json):
    job_title = jd_json.get("job_title", "").strip()
    skills = jd_json.get("skills_required", [])
    experience_str = jd_json.get("experience", "0 Years")
    experience_years = 0
    match = re.search(r"(\d+)\s*Years?", experience_str)
    if match:
        experience_years = int(match.group(1))
    return job_title, skills, experience_years

def create_prompt(skills, levels, question_count):
    skills_text = ', '.join(skills)
    levels_text = ', '.join(levels)

    if "High" in levels_text:
        difficulty = "Generate complex and tricky questions that require multi-step reasoning."
    else:
        difficulty = "Generate questions that test fundamental understanding."

    return f"""
Generate exactly {question_count} unique MCQ interview questions using a mix of the following skills and levels.

Skills: {skills_text}
Levels: {levels_text}

Instructions:
{difficulty}

- Each question must include:
    - Skill
    - Level
    - Question
    - 4 options labeled A) to D)
    - Correct answer (A/B/C/D)
- Format strictly like this (no extra text or explanation):

Skill: <skill>
Level: <Basic/Medium/High>
Question: <question text>
A) <option1>
B) <option2>
C) <option3>
D) <option4>
Answer: <A/B/C/D>

Start now.
"""

def generate_batch(skills, levels, question_count=15):
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    prompt = create_prompt(skills, levels, question_count)
    response = model.generate_content(prompt)
    all_text = response.text
    generated_questions = all_text.count("Question:")
    while generated_questions < question_count:
        additional = model.generate_content(create_prompt(skills, levels, question_count - generated_questions))
        all_text += additional.text
        generated_questions = all_text.count("Question:")
    return all_text

def generate_full_mcqs(skills, levels):
    batch1 = generate_batch(skills, levels, 15)
    batch2 = generate_batch(skills, levels, 15)
    df = parse_questions_to_df(batch1 + "\n\n" + batch2)
    return df

def parse_questions_to_df(text):
    pattern = r"Skill: (.*?)\nLevel: (.*?)\nQuestion: (.*?)\nA\) (.*?)\nB\) (.*?)\nC\) (.*?)\nD\) (.*?)\nAnswer: (.*?)\n"
    matches = re.findall(pattern, text, re.DOTALL)
    rows = []
    for match in matches:
        correct_letter = match[7].strip().upper()
        correct_answer = match[3:7][ord(correct_letter) - 65] if correct_letter in "ABCD" else "Unknown"
        rows.append({
            "skill": match[0].strip(),
            "skill_level": match[1].strip(),
            "question": match[2].strip(),
            "option1": match[3].strip(),
            "option2": match[4].strip(),
            "option3": match[5].strip(),
            "option4": match[6].strip(),
            "correct_answer": correct_answer
        })
    return pd.DataFrame(rows)

@app.post("/generate_mcqs/")
async def generate_mcqs(jd: JobDescription):
    try:
        job_title, skills, experience_years = extract_details_from_jd(jd.dict())
        levels = ["Basic", "Medium"] if experience_years <= 2 else ["Medium"] if experience_years <= 5 else ["High"]
        df = generate_full_mcqs(skills, levels)
        return {"mcqs": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating MCQs: {str(e)}")

# ------------------------ SALARY PREDICTION ------------------------
def build_prompt(data: JobInput) -> str:
    return (
        f"You are a salary prediction engine.\n"
        f"Your ONLY task is to return an estimated annual salary range in **Indian Rupees (INR)**.\n"
        f"DO NOT explain anything. DO NOT add any extra text. DO NOT use bullet points. JUST output the salary range.\n"
        f"Always return the salary range in this format exactly: 8,00,000–12,00,000 INR\n\n"
        f"Job Title: {data.job_title}\n"
        f"Location: {data.location}\n"
        f"Experience: {data.experience}\n"
        f"Skills: {data.skills}\n"
        f"Job Description: {data.job_description}\n"
    )

@app.post("/predict_salary/")
async def predict_salary(data: JobInput):
    try:
        prompt = build_prompt(data)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        return {"predicted_salary_range": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
