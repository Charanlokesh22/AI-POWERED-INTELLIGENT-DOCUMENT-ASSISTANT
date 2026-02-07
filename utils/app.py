from fastapi import FastAPI, UploadFile, File
from utils.rag_pipeline import add_documents, generate_answer

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = text.split("\n")
    add_documents(chunks)
    return {"message": "Document processed"}

@app.get("/ask")
def ask(question: str):
    answer = generate_answer(question)
    return {"answer": answer}
