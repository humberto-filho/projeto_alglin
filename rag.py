import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODELO = SentenceTransformer("all-MiniLM-L6-v2")

def cos_sim(a, b):  
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    dot = a @ b.T
    cos_sim = dot / (a_norm * b_norm)

    return cos_sim.ravel()

def carregar_receitas(path):
    with open(path, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados["recipes"]

def texto_para_busca(receita):
    titulo = receita.get("title")
    ingredientes = receita.get("ingredients", [])
    ner = receita.get("ner", [])
    texto = receita.get("text")
    
    partes = [str(titulo), str(ingredientes), str(ner), str(texto)]
    return " ".join(p for p in partes if p).strip()

def gerar_embeddings(receitas):
    textos = [texto_para_busca(r) for r in receitas]
    modelo = MODELO

    embeddings = modelo.encode(textos)
    return embeddings

def buscar(query, embeddings, receitas, top_k=5):
    modelo = MODELO
    query_embedding = modelo.encode([query])

    similaridades = cos_sim(query_embedding, embeddings)
    indices = np.argsort(similaridades)[::-1][:top_k]

    resultados = []
    for idx in indices:
        receita = receitas[idx].copy()
        receita["score"] = float(similaridades[idx])
        resultados.append(receita)
    return resultados

def main():
    query = input("Qual receita você quer procurar? (inglês)")

    print("Carregando o modelo de criar embeddings:")
    modelo = MODELO

    receitas = carregar_receitas("receitas.json")
    embeddings = gerar_embeddings(receitas)

    resultados = buscar(query, embeddings, receitas)

    print(f'Resultados para: {query}')
    for i, r in enumerate(resultados, 1):
        print(f"{i} Título: {r['title']} (score: {r['score']:.4f})")
        print(f"Ingredientes: {r['ingredients']}")


if __name__ == "__main__":
    main()