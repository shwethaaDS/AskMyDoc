# vectorstore.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunk_texts = []

    def build_index(self, embeddings, chunks):
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunk_texts = chunks

    def query(self, embedding, top_k=3):
        D, I = self.index.search(np.array([embedding]).astype("float32"), top_k)
        return [self.chunk_texts[i] for i in I[0]]
