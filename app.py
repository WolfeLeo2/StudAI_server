import sentencepiece  
import logging
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from queue_manager import RequestQueue
from config import MAX_QUEUE_SIZE, REQUEST_TIMEOUT

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "google/flan-t5-small"  # or "google/flan-t5-base" if small is too limited

# Load model and tokenizer
#logger.info(f"Loading model: {model_name}")
#tokenizer = T5Tokenizer.from_pretrained(model_name)
#model = T5ForConditionalGeneration.from_pretrained(model_name)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#ogger.info(f"Model loaded on device: {device}")

# Add this after your FastAPI app initialization
request_queue = RequestQueue(max_queue_size=MAX_QUEUE_SIZE, timeout=REQUEST_TIMEOUT)

class QuestionAnswerRequest(BaseModel):
    question: str
    context: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    try:
        logger.info(f"Loading model: {model_name}")
        tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16  # Use half precision
        )
        model.to(device)
        logger.info(f"Model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model
    del tokenizer
    import gc
    gc.collect()

app = FastAPI(lifespan=lifespan)

@app.post("/question-answer")
async def answer_question(request: QuestionAnswerRequest):
    try:
        answer = await request_queue.enqueue_request(
            async_answer_question,
            request.question,
            request.context
        )
        return {"answer": answer}
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this new async function for question answering
async def async_answer_question(question: str, context: str):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
        num_beams=4,  # Reduced from 6
        repetition_penalty=2.0,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False  # Deterministic generation uses less memory
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated summary of length {len(summary)}")
    return summary

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    try:
        summary = await request_queue.enqueue_request(
            summarize_text,
            request.text,
            request.max_length,
            request.min_length
        )
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ---------- Entry Point ----------

#if __name__ == "__main__":
 #   import uvicorn
 #   uvicorn.run(app, host="0.0.0.0", port=7860)
