from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_name = "Salesforce/codegen-350M-multi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained("riscv_model")

# Set up the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize FastAPI
app = FastAPI(title="RISC-V Code Generation API", version="1.0")

# Input schema
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30

# Root route
@app.get("/")
def read_root():
    return {"message": "RISC-V CodeGen API is running"}

# Code generation endpoint
@app.post("/generate")
def generate_code(req: PromptRequest):
    result = pipe(req.prompt, max_new_tokens=req.max_new_tokens)[0]["generated_text"]
    return {"generated_code": result}
