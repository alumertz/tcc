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
results_dir = "results/omics"
output_dir = "results/plots/omics"
os.makedirs(output_dir, exist_ok=True)

# Modelos que serão avaliados
model_names = [
    'decision_tree', 'random_forest', 'gradient_boosting',
    'histogram_gradient_boosting', 'k_nearest_neighbors', 'multi_layer_perceptron', # 'svc'
]

# Loop por modelo
for model_name in model_names:
    model_path = os.path.join(results_dir, model_name)
    if not os.path.exists(model_path):
        print(f"[AVISO] Diretório não encontrado: {model_path}")
        continue

    # Coletar todos os arquivos JSON de trials
    json_files = sorted([f for f in os.listdir(model_path) if f.startswith("trials_") and f.endswith(".json")])
    if not json_files:
        print(f"[AVISO] Nenhum trials_*.json encontrado para {model_name}")
        continue

    # Lista para armazenar scores deste modelo
    model_scores = []

    for json_file in json_files:
        json_path = os.path.join(model_path, json_file)
        try:
            with open(json_path, "r") as f:
                trials = json.load(f)
                for trial in trials:
                    score = trial.get("score")
                    if score is not None:
                        model_scores.append({
                            "Arquivo": json_file,
                            "Score": score
                        })
        except Exception as e:
            print(f"[ERRO] Falha ao ler {json_path}: {e}")
            continue

    # Verifica se há dados
    if not model_scores:
        print(f"[AVISO] Nenhum score encontrado para {model_name}")
        continue

    # Criar DataFrame
    df_model = pd.DataFrame(model_scores)

    # Criar boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Arquivo", y="Score", data=df_model, palette="Set2")
    sns.stripplot(x="Arquivo", y="Score", data=df_model, color="black", size=4, jitter=True, alpha=0.6)

    model_display_name = model_name.replace('_', ' ').title()
    plt.title(f"Distribuição dos Scores - {model_display_name}", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Arquivo JSON", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salvar gráfico
    safe_model_name = model_name.replace(" ", "_")
    output_path = os.path.join(output_dir, f"boxplot_{safe_model_name}_scores")
    plt.savefig(output_path + ".png", bbox_inches='tight')
    plt.savefig(output_path + ".pdf", bbox_inches='tight')
    plt.close()

    print(f"[OK] Boxplot salvo para {model_name} → {output_path}.png e .pdf")

print("\n✅ Todos os gráficos foram gerados com sucesso.")
