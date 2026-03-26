from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional
from datetime import datetime, timezone
import sqlite3
import torch

# --- Database Setup ---

DB_PATH = "ratings.db"

# Initialize database if it doesn't exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rating (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prefix TEXT NOT NULL,
            suffix TEXT NOT NULL DEFAULT '',
            suggestion TEXT NOT NULL,
            rating INTEGER CHECK(rating = 0 OR rating = 1 OR rating IS NULL),
            suggestion_type TEXT NOT NULL DEFAULT 'realtime',
            accepted INTEGER NOT NULL DEFAULT 1,
            timestamp TEXT NOT NULL
        )
    ''')
    # Migrate existing tables that are missing new columns
    for col, definition in [
        ('suffix', "TEXT NOT NULL DEFAULT ''"),
        ('suggestion_type', "TEXT NOT NULL DEFAULT 'realtime'"),
        ('accepted', 'INTEGER NOT NULL DEFAULT 1'),
        ('timestamp', "TEXT NOT NULL DEFAULT ''"),
    ]:
        try:
            cursor.execute(f'ALTER TABLE rating ADD COLUMN {col} {definition}')
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()

# Call once on startup
init_db()

# Save rating to database
def save_rating(prefix: str, suffix: str, suggest: str, rating, suggestion_type: str, accepted: int, timestamp: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO rating (prefix, suffix, suggestion, rating, suggestion_type, accepted, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (prefix, suffix, suggest, rating, suggestion_type, accepted, timestamp)
    )
    conn.commit()
    conn.close()

# --- FastAPI Setup ---

# checkpoint_path = "trained_model/starcoder-riscv-fim-A100/checkpoint-525"
model_path = "trained_model/final_model"

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

def load_model(model_path):
    # `model_path` is a PEFT LoRA adapter directory (trained_model/final_model)
    base_model_name = "Salesforce/codegen-350M-multi"

    # 1) Load tokenizer FIRST (this vocab size is what your adapter expects)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2) Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=False,
    )

    # 3) Make base model embeddings match tokenizer vocab (fixes size mismatch 50298 vs 51200)
    base_model.resize_token_embeddings(len(tokenizer))

    # 4) Apply the LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer


def complete_code(model, tokenizer, prompt="", max_new_tokens=50):
    """Complete code using FIM format."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    prompt_decode = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    if generated.startswith(prompt_decode):
        completion = generated[len(prompt_decode):]
        # print(completion)
    else:
        completion = generated
        print(completion)

    return completion

model, tokenizer = load_model(model_path)

# Set up the pipeline
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize FastAPI
app = FastAPI(title="RISC-V Code Generation API", version="1.0")

# Input Schemas
class PromptRequest(BaseModel):
    prefix: str
    suffix: str = ''
    max_new_tokens: int = 30

class RatingRequest(BaseModel):
    prefix: str
    suffix: str = ''
    suggestion: str
    rating: Optional[int] = None
    suggestion_type: str = 'realtime'
    accepted: bool = True
    timestamp: str = ''

# Root route
@app.get("/")
def read_root():
    return {"message": "RISC-V CodeGen API is running"}

# Code generation endpoint
@app.post("/generate")
def generate_code(req: PromptRequest):
    if not req.prefix:
        return {"error": "Prefix is required"}
    if req.max_new_tokens <= 0:
        return {"error": "max_new_tokens must be greater than 0"}
    if len(req.prefix) > 512:
        return {"error": "Prefix length exceeds 512 characters"}

    prompt = f"{FIM_PREFIX}{req.prefix}{FIM_SUFFIX}{req.suffix}{FIM_MIDDLE}"
    result = complete_code(model, tokenizer, prompt, max_new_tokens=req.max_new_tokens)
    return {"generated_code": result}

# Rating endpoint with DB insert
@app.post("/rating")
def rating_code(req: RatingRequest):
    if not req.prefix:
        return {"error": "Prefix is required"}
    if not req.suggestion:
        return {"error": "Suggestion is required"}
    if req.rating is not None and req.rating not in (0, 1):
        return {"error": "Rating must be 0 or 1"}
    if req.suggestion_type not in ('realtime', 'comment-to-code'):
        return {"error": "suggestion_type must be 'realtime' or 'comment-to-code'"}

    ts = req.timestamp or datetime.now(timezone.utc).isoformat()
    save_rating(req.prefix, req.suffix, req.suggestion, req.rating, req.suggestion_type, int(req.accepted), ts)
    return {"message": "Rating saved"}
