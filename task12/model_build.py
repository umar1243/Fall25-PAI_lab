import pandas as pd
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.lower()
    return ''

df = pd.read_csv("habits_data.csv")
df['clean_question'] = df['question'].apply(clean_text)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(df['clean_question'].tolist())

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))

faiss.write_index(index, "habits.index")

print("Index created successfully!")
