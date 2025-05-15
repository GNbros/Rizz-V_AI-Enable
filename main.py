from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sqlite3

# --- Database Setup ---

DB_PATH = "ratings.db"

# Initialize database if it doesn't exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rating (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5)
        )
    ''')
    conn.commit()
    conn.close()

# Call once on startup
init_db()

# Save rating to database
def save_rating(prompt: str, suggest: str ,rating: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO rating (prompt, suggestion, rating) VALUES (?, ?, ?)", (prompt, suggest, rating))
    conn.commit()
    conn.close()

# --- FastAPI Setup ---

# Load model and tokenizer
model_name = "Salesforce/codegen-350M-multi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained("riscv_model")

# Set up the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize FastAPI
app = FastAPI(title="RISC-V Code Generation API", version="1.0")

# Input Schemas
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30

class RatingRequest(BaseModel):
    prompt: str
    suggestion: str
    rating: int

# Root route
@app.get("/")
def read_root():
    return {"message": "RISC-V CodeGen API is running"}

# Code generation endpoint
@app.post("/generate")
def generate_code(req: PromptRequest):
    if not req.prompt:
        return {"error": "Prompt is required"}
    if req.max_new_tokens <= 0:
        return {"error": "max_new_tokens must be greater than 0"}
    if len(req.prompt) > 512:
        return {"error": "Prompt length exceeds 512 characters"}

    result = pipe(req.prompt, max_new_tokens=req.max_new_tokens)[0]["generated_text"]
    return {"generated_code": result}

# Rating endpoint with DB insert
@app.post("/rating")
def rating_code(req: RatingRequest):
    if not req.prompt:
        return {"error": "Prompt is required"}
    if not req.suggestion:
        return {"error": "Suggestion is required"}
    if req.rating < 1 or req.rating > 5:
        return {"error": "Rating must be between 1 and 5"}

    save_rating(req.prompt,req.suggestion, req.rating)
    return {"message": "Rating saved", "rating": req.rating, "prompt": req.prompt}
