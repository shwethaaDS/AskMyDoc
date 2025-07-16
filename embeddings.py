from sentence_transformers import SentenceTransformer


model = SentenceTransformer("*****HUGGING FACE MODEL*******")  

def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=True).tolist()
