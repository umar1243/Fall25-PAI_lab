from flask import Flask, render_template, request
import pandas as pd
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.lower()
    return ''

df = pd.read_csv("habits_data.csv")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
index = faiss.read_index("habits.index")

def get_answer(query, k=1):
    query = clean_text(query)
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)
    result = df.iloc[indices[0][0]]
    return result['answer']

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = get_answer(question)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = get_answer(question)
    return render_template("index.html", answer=answer)