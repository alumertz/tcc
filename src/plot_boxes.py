#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 🔧 CONFIGURAÇÕES DE VISUALIZAÇÃO
# ==============================
sns.set(style="whitegrid", context="talk", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ==============================
# 📂 DIRETÓRIOS
# ==============================
results_dir = "/home/kamikp/Documents/tcc/results/omics"
output_dir = "/home/kamikp/Documents/tcc/results/plots/omics"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# ⚙️ MODELOS
# ==============================
model_names = [
    'decision_tree', 'random_forest', 'gradient_boosting', 'histogram_gradient_boosting',
    'k_nearest_neighbors', 'multi_layer_perceptron', 'support_vector_classifier', 'catboost'
]

all_scores = []

# ==============================
# 📥 LEITURA DOS RESULTADOS
# ==============================
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
                        "Feature_Variation": json_file
                    })
        except Exception as e:
            print(f"[ERRO] Falha ao ler {json_path}: {e}")
            continue

# ==============================
# 📊 PLOTAGEM DO BOXPLOT
# ==============================
if not all_scores:
    print("[AVISO] Nenhum score encontrado para nenhum modelo.")
else:
    df_all = pd.DataFrame(all_scores)

    # Ordenar os modelos pela mediana dos scores
    order = df_all.groupby("Modelo")["Score"].median().sort_values(ascending=False).index

    plt.figure(figsize=(12, 8))

    # Boxplot principal
    sns.boxplot(
        x="Modelo", y="Score", data=df_all,
        order=order, palette="pastel",
        width=0.6, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
    )

    # Pontos individuais (distribuição)
    sns.stripplot(
        x="Modelo", y="Score", data=df_all,
        order=order, color="black", size=4,
        jitter=True, alpha=0.6
    )

    # Títulos e rótulos
    plt.title("Distribuição dos Scores por Modelo (Variações de Features)", pad=20)
    plt.ylabel("Accuracy Global")
    plt.xlabel("Modelo")

    # Melhor rotação dos rótulos
    plt.xticks(rotation=30, ha='right')

    # Grade leve e limites ajustados
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Estatísticas anotadas acima das caixas (mediana)
    medians = df_all.groupby("Modelo")["Score"].median()
    for tick, label in enumerate(order):
        median_val = medians[label]
        plt.text(
            tick, median_val + 0.005, f"{median_val:.3f}",
            horizontalalignment='center', color='black', weight='semibold', fontsize=10
        )

    plt.tight_layout()

    # ==============================
    # 💾 SALVAMENTO
    # ==============================
    output_path = os.path.join(output_dir, "boxplot_scores_por_modelo")
    plt.savefig(output_path + ".png", bbox_inches='tight')
    plt.savefig(output_path + ".pdf", bbox_inches='tight')
    plt.close()

    print(f"[OK] Boxplot salvo → {output_path}.png e .pdf")

print("\n✅ Todos os gráficos foram gerados com sucesso.")
