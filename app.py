from fastapi import FastAPI, UploadFile, File, Body
from extract_text import extract_text
from embeddings import get_embeddings
from vectorstore import VectorStore
from llm import generate_answer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

vector_store = VectorStore()
stored_chunks = []


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    saved_filename = f"uploaded_{file.filename}"

    
    with open(saved_filename, "wb") as f:
        f.write(contents)


    extracted_text = extract_text(saved_filename)

    
    chunks = extracted_text.split("\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  

   
    sample_chunks = chunks[:5]


    vectors = get_embeddings(sample_chunks)

    vector_store.build_index(vectors, sample_chunks)
    stored_chunks = sample_chunks


    return {
        "filename": file.filename,
        "message": "File uploaded, text extracted, and embeddings generated.",
        "extracted_text_preview": extracted_text[:500],
        "embeddings": vectors
    }

@app.post("/chat")
async def chat(question: str = Body(...)):
    question_embedding = get_embeddings([question])[0]
    top_chunks = vector_store.query(question_embedding, top_k=3)

    context = "\n".join(top_chunks)
    prompt = f"""You are a helpful assistant. provide relevant Answer to the question ONLY based on the document excerpt.
If the answer is not contained in the document, respond with "I don't know".

Document:
{context}

Question: {question}
Answer:"""

    answer = generate_answer(prompt)

    return {
        "question": question,
        "context": context,
        "answer": answer
    }
