#!/usr/bin/env python3

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações de visualização
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Diretórios
results_dir = "/home/kamikp/Documents/tcc/results/omics"
output_dir = "/home/kamikp/Documents/tcc/results/plots/omics"
os.makedirs(output_dir, exist_ok=True)

# Modelos que serão avaliados
model_names = [
    'decision_tree', 'random_forest', 'gradient_boosting', 'histogram_gradient_boosting', 
    'k_nearest_neighbors', 'multi_layer_perceptron', 'support_vector_classifier', 'catboost'
]

all_scores = []

for model_name in model_names:
    model_path = os.path.join(results_dir, model_name)
    if not os.path.exists(model_path):
        print(f"[AVISO] Diretório não encontrado: {model_path}")
        continue

    json_files = sorted([f for f in os.listdir(model_path) if f.endswith(".json")])
    if not json_files:
        print(f"[AVISO] Nenhum arquivo .json encontrado para {model_name}")
        continue

    for json_file in json_files:
        json_path = os.path.join(model_path, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                accuracy = data.get("accuracy_global")
                if accuracy is not None:
                    all_scores.append({
                        "Modelo": model_name.replace('_', ' ').title(),
                        "Score": accuracy,
                        "Feature_Variation": json_file  # opcional, para controle
                    })
        except Exception as e:
            print(f"[ERRO] Falha ao ler {json_path}: {e}")
            continue

if not all_scores:
    print("[AVISO] Nenhum score encontrado para nenhum modelo.")
else:
    df_all = pd.DataFrame(all_scores)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Modelo", y="Score", data=df_all, palette="Set3")
    sns.stripplot(x="Modelo", y="Score", data=df_all, color="black", size=4, jitter=True, alpha=0.6)

    plt.title("Distribuição dos Scores por Modelo (variações de features)", fontsize=16)
    plt.ylabel("Accuracy Global", fontsize=14)
    plt.xlabel("Modelo", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = os.path.join(output_dir, "boxplot_scores_por_modelo")
    plt.savefig(output_path + ".png", bbox_inches='tight')
    plt.savefig(output_path + ".pdf", bbox_inches='tight')
    plt.close()

    print(f"[OK] Boxplot geral salvo → {output_path}.png e .pdf")


'''all_scores = []


for model_name in model_names:
    model_path = os.path.join(results_dir, model_name)
    if not os.path.exists(model_path):
        print(f"[AVISO] Diretório não encontrado: {model_path}")
        continue

    json_files = sorted([f for f in os.listdir(model_path) if f.endswith(".json")])
    if not json_files:
        print(f"[AVISO] Nenhum arquivo .json encontrado para {model_name}")
        continue

    for json_file in json_files:
        json_path = os.path.join(model_path, json_file)
        try:
            with open(json_path, "r") as f:
                trials = json.load(f)
                accuracy = trials.get("accuracy_global")
                if accuracy is not None:
                    all_scores.append({
                        "Modelo": model_name.replace('_', ' ').title(),
                        "Score": accuracy
                    })
        except Exception as e:
            print(f"[ERRO] Falha ao ler {json_path}: {e}")
            continue

if not all_scores:
    print("[AVISO] Nenhum score encontrado para nenhum modelo.")
else:
    df_all = pd.DataFrame(all_scores)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Modelo", y="Score", data=df_all, palette="Set3")
    sns.stripplot(x="Modelo", y="Score", data=df_all, color="black", size=4, jitter=True, alpha=0.6)

    plt.title("Distribuição dos Scores por Modelo", fontsize=16)
    plt.ylabel("Accuracy Global", fontsize=14)
    plt.xlabel("Modelo", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = os.path.join(output_dir, "boxplot_scores_por_modelo")
    plt.savefig(output_path + ".png", bbox_inches='tight')
    plt.savefig(output_path + ".pdf", bbox_inches='tight')
    plt.close()

    print(f"[OK] Boxplot geral salvo → {output_path}.png e .pdf")'''

print("\n✅ Todos os gráficos foram gerados com sucesso.")
