from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
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

# checkpoint_path = "trained_model/starcoder-riscv-fim-A100/checkpoint-525"
model_path = "trained_model/final_model"

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

def load_model(model_path):
    # `model_path` is a PEFT LoRA adapter directory (trained_model/final_model)
    base_model_name = "Salesforce/codegen-350M-mono"

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
    
    prompt = f"{FIM_PREFIX}{req.prompt}{FIM_SUFFIX}{FIM_MIDDLE}"
    result = complete_code(model, tokenizer, prompt, max_new_tokens=req.max_new_tokens)
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
