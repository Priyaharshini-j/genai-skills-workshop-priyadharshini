from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import fetch_faq_results, generate_bot_response

app = FastAPI()

class QueryRequest(BaseModel):
    question: str


@app.post("/chat")
def ask_question(req: QueryRequest):
    context_df = fetch_faq_results(req.question)
    response = generate_bot_response(req.question)
    return {"answer": response.text.strip()}
