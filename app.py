import sentencepiece  
import logging
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "google/flan-t5-large"

# Load model and tokenizer
logger.info(f"Loading model: {model_name}")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Model loaded on device: {device}")


class QuestionAnswerRequest(BaseModel):
    question: str
    context: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, tokenizer
    try:
        logger.info(f"Loading model: {model_name}")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        logger.info(f"Model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # Shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/question-answer")
async def answer_question(request: QuestionAnswerRequest):
    try:
        input_text = f"question: {request.question} context: {request.context}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(
            inputs.input_ids, 
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150
    min_length: Optional[int] = 30

def summarize_text(text, max_length=150, min_length=30):
    logger.info(f"Summarizing text of length {len(text)}")
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512).to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        min_length=min_length,
        num_beams=6,
        repetition_penalty=2.0,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated summary of length {len(summary)}")
    return summary

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    try:
        summary = summarize_text(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ---------- Entry Point ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
