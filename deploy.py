import os
import json
import re
import io
import asyncio
import logging
import time
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, validator
import google.generativeai as genai
import uvicorn

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hr_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global Metrics Tracking
# -----------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.active_requests = 0
        self.total_requests = 0
        self.error_count = 0
        self.start_time = time.time()
        self.endpoint_stats = {}
    
    def request_started(self, endpoint: str):
        self.active_requests += 1
        self.total_requests += 1
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {"count": 0, "errors": 0}
        self.endpoint_stats[endpoint]["count"] += 1
    
    def request_completed(self):
        self.active_requests = max(0, self.active_requests - 1)
    
    def error_occurred(self, endpoint: str = None):
        self.error_count += 1
        if endpoint and endpoint in self.endpoint_stats:
            self.endpoint_stats[endpoint]["errors"] += 1
    
    def get_stats(self):
        uptime = time.time() - self.start_time
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "uptime_seconds": round(uptime, 2),
            "requests_per_minute": round((self.total_requests / uptime) * 60, 2) if uptime > 0 else 0,
            "endpoint_stats": self.endpoint_stats
        }

metrics = MetricsTracker()

# -----------------------------------------------------------------------------
# Application Lifespan Events
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 HR/Recruitment API starting up...")
    logger.info(f"📊 Workers: {os.getenv('WORKERS', '4')}")
    logger.info(f"🔑 Gemini API configured: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
    logger.info(f"⚡ Max concurrent requests: {os.getenv('MAX_CONCURRENT_REQUESTS', '50')}")
    yield
    # Shutdown
    logger.info("🛑 HR/Recruitment API shutting down...")
    logger.info(f"📈 Final stats: {metrics.get_stats()}")

# -----------------------------------------------------------------------------
# Enhanced Gemini API Manager
# -----------------------------------------------------------------------------
class GeminiAPIManager:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBrc0vyseVt5Ed3-BK7jobOAN4I12R1E8Q")
        genai.configure(api_key=api_key)
        
        self.model_name = "gemini-2.0-flash-lite"
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '50'))
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.request_count = 0
        self.active_requests = 0
        
        logger.info(f"🎯 Gemini API Manager initialized with {self.max_concurrent_requests} concurrent slots")
    
    async def generate_content_async(self, prompt: str, timeout: int = 45, request_id: str = None) -> str:
        """Enhanced async wrapper for Gemini API with detailed logging"""
        request_id = request_id or str(uuid.uuid4())[:8]
        
        async with self.semaphore:
            self.active_requests += 1
            self.request_count += 1
            
            logger.info(f"🤖 [{request_id}] Starting Gemini API call ({self.active_requests}/{self.max_concurrent_requests} active)")
            
            try:
                start_time = time.time()
                
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._sync_generate, prompt, request_id),
                    timeout=timeout
                )
                
                processing_time = time.time() - start_time
                logger.info(f"✅ [{request_id}] Gemini API completed in {processing_time:.2f}s")
                
                return response
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ [{request_id}] Gemini API timeout after {timeout}s")
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"AI processing timeout after {timeout} seconds"
                )
            except Exception as e:
                logger.error(f"❌ [{request_id}] Gemini API error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"AI service temporarily unavailable: {str(e)}"
                )
            finally:
                self.active_requests -= 1
    
    def _sync_generate(self, prompt: str, request_id: str) -> str:
        """Synchronous Gemini API call"""
        try:
            model = genai.GenerativeModel(self.model_name)
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            logger.error(f"🔥 [{request_id}] Sync generation error: {str(e)}")
            raise e
    
    def get_status(self):
        """Get current API manager status"""
        return {
            "max_concurrent": self.max_concurrent_requests,
            "active_requests": self.active_requests,
            "available_slots": self.max_concurrent_requests - self.active_requests,
            "total_requests": self.request_count,
            "utilization_percent": round((self.active_requests / self.max_concurrent_requests) * 100, 1)
        }

# Initialize Gemini manager
gemini_manager = GeminiAPIManager()

# -----------------------------------------------------------------------------
# Enhanced Pydantic Models with Validation
# -----------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    jd: Dict[str, Any]
    resume: Dict[str, Any]
    
    @validator('jd', 'resume')
    def validate_data_size(cls, v):
        if len(json.dumps(v)) > 100000:  # 100KB limit
            raise ValueError('Data too large (max 100KB)')
        return v

class QuestionRequest(BaseModel):
    resume: Dict[str, Any]
    jd: Optional[Dict[str, Any]] = None
    
    @validator('resume', 'jd')
    def validate_data_size(cls, v):
        if v and len(json.dumps(v)) > 75000:
            raise ValueError('Data too large (max 75KB)')
        return v

class JobDescription(BaseModel):
    job_title: str
    skills_required: list
    experience: str
    
    @validator('job_title')
    def validate_job_title(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Job title must be at least 2 characters')
        if len(v) > 200:
            raise ValueError('Job title too long (max 200 characters)')
        return v.strip()
    
    @validator('skills_required')
    def validate_skills(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one skill is required')
        if len(v) > 50:
            raise ValueError('Too many skills (max 50)')
        return v

class JobInput(BaseModel):
    job_title: str
    location: List[str]
    job_description: str
    skills: List[str]
    experience: str
    
    @validator('job_title', 'job_description', 'experience')
    def validate_text_fields(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Field must be at least 2 characters')
        return v.strip()
    
    @validator('location', 'skills')
    def validate_lists(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one item is required')
        return v

# -----------------------------------------------------------------------------
# FastAPI Setup with Enhanced Configuration
# -----------------------------------------------------------------------------
app = FastAPI(
    title="HR_RECRUITMENT_API_V2",
    description="High-performance parallel processing HR and recruitment tools with AI analysis",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# -----------------------------------------------------------------------------
# Enhanced Middleware for Request Tracking
# -----------------------------------------------------------------------------
@app.middleware("http")
async def enhanced_request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    start_time = time.time()
    endpoint = request.url.path
    metrics.request_started(endpoint)
    
    logger.info(f"📥 [{request_id}] {request.method} {endpoint} - "
               f"Client: {request.client.host} - Active: {metrics.active_requests}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(f"✅ [{request_id}] {request.method} {endpoint} - "
                   f"{response.status_code} - {process_time:.3f}s")
        
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Active-Requests"] = str(metrics.active_requests)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        metrics.error_occurred(endpoint)
        
        logger.error(f"❌ [{request_id}] {request.method} {endpoint} - "
                    f"Error: {str(e)[:100]} - {process_time:.3f}s")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred during processing",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path)
            },
            headers={
                "X-Process-Time": f"{process_time:.3f}",
                "X-Request-ID": request_id
            }
        )
    finally:
        metrics.request_completed()

# -----------------------------------------------------------------------------
# Health Check and Monitoring Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint with API status"""
    return {
        "message": "HR/Recruitment FastAPI is alive!",
        "version": "2.1.0",
        "parallel_processing": "enabled",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "hr-recruitment-api",
        "version": "2.1.0",
        "parallel_processing": "enabled"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with performance metrics"""
    gemini_status = gemini_manager.get_status()
    app_metrics = metrics.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "hr-recruitment-api",
        "version": "2.1.0",
        "parallel_processing": {
            "enabled": True,
            "gemini_api_status": gemini_status
        },
        "performance_metrics": app_metrics,
        "system_info": {
            "workers": os.getenv("WORKERS", "4"),
            "max_concurrent_requests": gemini_manager.max_concurrent_requests
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Detailed metrics endpoint for monitoring"""
    return {
        "application_metrics": metrics.get_stats(),
        "gemini_api_metrics": gemini_manager.get_status(),
        "timestamp": datetime.utcnow().isoformat()
    }

# -----------------------------------------------------------------------------
# Async Processing Functions
# -----------------------------------------------------------------------------
async def analyze_candidate_async(jd: dict, resume: dict, request_id: str = None) -> str:
    """Async candidate analysis with parallel processing support"""[1]
    prompt = f"""
    You are an AI assistant that helps recruiters analyze how well a candidate fits a job based on the Job Description (JD) and Resume.

    **Your Tasks:**
    1. Identify the required skills from the JD (from the skills key only).
    2. Look for those skills **only in these resume sections**:
       - skillsDetails.skills
       - professionalDetails[*].skills
       - projectDetails[*].skills
    3. If a skill appears in any of the above, add it under **Matching Skills**.
    4. If a skill does NOT appear in any of the above, add it under **Non-Matching Skills**.

    **Important Notes:**
    - Do NOT say "Recruitment (Not Found)" — just list matching skills under one heading, and missing ones under the other.
    - If there are **no matching skills**, write **"No Matching Skills"** under that section.
    - Your final summary must be clear, in detailed, structured, and in **bullet points**, not paragraph form.
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
    - **Interview is not recommended** for this role and also write why.
    """
    
    return await gemini_manager.generate_content_async(prompt, timeout=45, request_id=request_id)

async def generate_interviewer_questions_async(resume: dict, jd: dict = None, request_id: str = None) -> str:
    """Async interview question generation"""
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
    - For technical roles: prioritize domain-specific 70% technical questions 30% Non-technical questions.
    - For non-technical roles: include a mix of situational and behavioral questions.

    ✅ Format:
    - Use clear section headers as needed (Technical, Non Technical) for technical roles only
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
    
    return await gemini_manager.generate_content_async(prompt, timeout=50, request_id=request_id)

async def generate_batch_async(skills: list, levels: list, question_count: int = 15, request_id: str = None) -> str:
    """Async batch MCQ generation"""
    skills_text = ', '.join(skills)
    levels_text = ', '.join(levels)
    
    if "High" in levels_text:
        difficulty = "Generate complex and tricky questions that require multi-step reasoning."
    else:
        difficulty = "Generate questions that test fundamental understanding."

    prompt = f"""
    Generate exactly {question_count} unique MCQ interview questions using a mix of the following skills and levels.

    Skills: {skills_text}
    Levels: {levels_text}

    Instructions: {difficulty}
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
    
    return await gemini_manager.generate_content_async(prompt, timeout=60, request_id=request_id)

async def generate_full_mcqs_async(skills: list, levels: list, request_id: str = None) -> pd.DataFrame:
    """Generate MCQs with parallel batch processing"""
    # Generate two batches concurrently for better performance
    batch1_task = generate_batch_async(skills, levels, 15, f"{request_id}-b1")
    batch2_task = generate_batch_async(skills, levels, 15, f"{request_id}-b2")
    
    # Wait for both batches to complete
    batch1, batch2 = await asyncio.gather(batch1_task, batch2_task)
    
    # Parse combined results
    combined_text = batch1 + "\n\n" + batch2
    df = parse_questions_to_df(combined_text)
    
    return df

def parse_questions_to_df(text: str) -> pd.DataFrame:
    """Parse MCQ text into DataFrame"""
    pattern = r"Skill: (.*?)\nLevel: (.*?)\nQuestion: (.*?)\nA\) (.*?)\nB\) (.*?)\nC\) (.*?)\nD\) (.*?)\nAnswer: (.*?)\n"
    matches = re.findall(pattern, text, re.DOTALL)
    
    rows = []
    for match in matches:
        correct_letter = match[7].strip().upper()
        if correct_letter in "ABCD":
            correct_answer = match[3:7][ord(correct_letter) - 65]
        else:
            correct_answer = "Unknown"
        
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

def extract_details_from_jd(jd_json: dict):
    """Extract job details from JD JSON"""
    job_title = jd_json.get("job_title", "").strip()
    skills = jd_json.get("skills_required", [])
    experience_str = jd_json.get("experience", "0 Years")
    
    experience_years = 0
    match = re.search(r"(\d+)\s*Years?", experience_str)
    if match:
        experience_years = int(match.group(1))
    
    return job_title, skills, experience_years

async def predict_salary_async(data: JobInput, request_id: str = None) -> str:
    """Async salary prediction"""
    prompt = (
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
    
    return await gemini_manager.generate_content_async(prompt, timeout=30, request_id=request_id)

# -----------------------------------------------------------------------------
# Enhanced Async Endpoints
# -----------------------------------------------------------------------------
@app.post("/CANDIDATE_ANALYSIS")
async def analyser(
    request: AnalyzeRequest,
    req: Request,
    x_forwarded_for: str = Header(None)
):
    """Enhanced async candidate analysis endpoint"""
    request_id = getattr(req.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(f"👤 [{request_id}] Starting candidate analysis")
        
        result = await analyze_candidate_async(request.jd, request.resume, request_id)
        
        logger.info(f"✅ [{request_id}] Candidate analysis completed")
        
        return {
            "analysis": result,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_mode": "parallel"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Candidate analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/GENERATE_QUESTIONS")
async def generate_questions(
    request: QuestionRequest,
    req: Request,
    x_forwarded_for: str = Header(None)
):
    """Enhanced async question generation endpoint"""
    request_id = getattr(req.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(f"❓ [{request_id}] Starting question generation")
        
        result = await generate_interviewer_questions_async(
            request.resume, 
            request.jd, 
            request_id
        )
        
        logger.info(f"✅ [{request_id}] Question generation completed")
        
        return {
            "questions": result,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_mode": "parallel"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Question generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )

@app.post("/generate_mcqs/")
async def generate_mcqs(
    jd: JobDescription,
    req: Request
):
    """Enhanced async MCQ generation endpoint"""
    request_id = getattr(req.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(f"📝 [{request_id}] Starting MCQ generation")
        
        job_title, skills, experience_years = extract_details_from_jd(jd.dict())
        
        # Determine difficulty levels based on experience
        if experience_years <= 2:
            levels = ["Basic", "Medium"]
        elif experience_years <= 5:
            levels = ["Medium"]
        else:
            levels = ["High"]
        
        df = await generate_full_mcqs_async(skills, levels, request_id)
        
        logger.info(f"✅ [{request_id}] MCQ generation completed - {len(df)} questions")
        
        return {
            "mcqs": df.to_dict(orient="records"),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_mode": "parallel",
            "questions_count": len(df)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] MCQ generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MCQ generation failed: {str(e)}"
        )

@app.post("/predict_salary/")
async def predict_salary(
    data: JobInput,
    req: Request
):
    """Enhanced async salary prediction endpoint"""
    request_id = getattr(req.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(f"💰 [{request_id}] Starting salary prediction for: {data.job_title}")
        
        result = await predict_salary_async(data, request_id)
        
        logger.info(f"✅ [{request_id}] Salary prediction completed")
        
        return {
            "predicted_salary_range": result.strip(),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_mode": "parallel"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] Salary prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Salary prediction failed: {str(e)}"
        )

# -----------------------------------------------------------------------------
# Global Exception Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.warning(f"⚠️ [{request_id}] Validation error on {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        },
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"💥 [{request_id}] Unhandled exception on {request.url.path}: {str(exc)}")
    metrics.error_occurred(request.url.path)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during processing",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        },
        headers={"X-Request-ID": request_id}
    )

# -----------------------------------------------------------------------------
# Production Uvicorn Configuration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Production-optimized configuration
    config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "workers": int(os.getenv("WORKERS", 4)),
        "worker_class": "uvicorn.workers.UvicornWorker",
        "loop": "uvloop",
        "http": "httptools",
        "access_log": True,
        "log_level": "info",
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 120,
        "limit_max_requests": 1000,
        "limit_concurrency": 1000,
        "backlog": 2048
    }
    
    logger.info(f"🚀 Starting HR/Recruitment API with configuration: {config}")
    uvicorn.run(**config)
