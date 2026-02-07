import os
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

documents = []
embeddings = []
dimension = 1536

index = faiss.IndexFlatL2(dimension)

def add_documents(text_list):
    global documents, embeddings
    for text in text_list:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        embeddings.append(emb)
        documents.append(text)

    index.add(np.array(embeddings).astype("float32"))

def search(query, k=3):
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    D, I = index.search(np.array([query_emb]).astype("float32"), k)
    return [documents[i] for i in I[0]]

def generate_answer(query):
    context = search(query)
    prompt = f"Answer using context: {context}\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
