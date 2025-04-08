import os
import PyPDF2
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss

# Inicializando o Flask
app = Flask(__name__)

# Carregando o modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Estrutura para armazenar PDFs e suas embeddings
documents = []
embeddings = []

# Função para carregar PDFs e extrair texto
def load_pdfs(pdf_folder):
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            with open(os.path.join(pdf_folder, filename), 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + ' '
                documents.append(text)
                embeddings.append(model.encode(text))

# Carregar PDFs da pasta 'inputs'
load_pdfs('inputs')

# Criar índice de busca vetorial
embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    question_embedding = model.encode(question).reshape(1, -1)
    
    # Busca na matriz de embeddings
    D, I = index.search(question_embedding, k=1)
    response = documents[I[0][0]] if len(I[0]) > 0 else "Desculpe, não consegui encontrar uma resposta."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
