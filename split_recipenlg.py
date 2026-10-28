import csv
import json
import random

N_AMOSTRA = 10_000
TOTAL = 2_231_142 # total de elementos em RecipeNLG.csv
SEED = 333


def parse_lista(texto):
    itens, atual = [], []
    dentro = False       
    escape = False       
    for ch in texto:
        if escape:
            atual.append(ch)
            escape = False
        elif dentro and ch == "\\":
            escape = True
        elif ch == '"':
            if dentro:                       
                itens.append("".join(atual).strip())
                atual = []
            dentro = not dentro              
        elif dentro:
            atual.append(ch)
    return [x for x in itens if x]


# máscara aleatória: 10 mil posições sorteadas
random.seed(SEED) # para reproducibilidade
mascara = [1] * N_AMOSTRA + [0] * (TOTAL - N_AMOSTRA)
random.shuffle(mascara)

receitas = []
with open("RecipeNLG.csv", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    i_title, i_ing, i_dir, i_ner = (
        header.index(c) for c in ("title", "ingredients", "directions", "NER")
    )

    for ii, row in enumerate(reader):
        if ii >= TOTAL:
            break
        if mascara[ii] and row[i_title].strip():
            ing  = parse_lista(row[i_ing])
            dirs = parse_lista(row[i_dir])
            if ing and dirs:
                titulo = row[i_title].strip()
                receitas.append({
                    "id":          len(receitas),
                    "title":       titulo,
                    "ingredients": ing,
                    "directions":  dirs,
                    "ner":         [x.lower() for x in parse_lista(row[i_ner])],
                    "text":        titulo + ". " + " ".join(dirs),
                })

with open("receitas.json", "w", encoding="utf-8") as f:
    json.dump({"recipes": receitas}, f, ensure_ascii=False, indent=1)

print(f"pronto: receitas.json")