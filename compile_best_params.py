import re
import csv
import os
import pandas as pd

# --- FUNÇÕES DE SIMULAÇÃO (MOCK) ---
# Você DEVE remover esta função e as chamadas a ela se for usar em produção
# com um diretório de resultados real.
def setup_mock_environment():
    """Cria uma estrutura de diretórios e arquivos de simulação."""
    print("Configurando ambiente de simulação...")
    
    # Conteúdo simplificado do arquivo para teste
    file_content_dt = """
RESULTADOS DO MODELO: DECISION_TREE
================================================================================
...
FOLD 1 (loop externo):
Importância dos parâmetros (Optuna):
  max_depth: 0.3775
  max_features: 0.2394
  min_samples_split: 0.1510
  min_samples_leaf: 0.0962
  splitter: 0.0719
  criterion: 0.0639
...
FOLD 2 (loop externo):
Importância dos parâmetros (Optuna):
  max_depth: 0.6692
  min_samples_leaf: 0.2449
  splitter: 0.0273
  max_features: 0.0261
  min_samples_split: 0.0217
  criterion: 0.0107
...
"""
    file_content_knn = """
RESULTADOS DO MODELO: K_NEAREST_NEIGHBORS
================================================================================
...
FOLD 1 (loop externo):
Importância dos parâmetros (Optuna):
  n_neighbors: 0.4500
  metric: 0.2500
  weights: 0.1500
  algorithm: 0.1000
...
FOLD 2 (loop externo):
Importância dos parâmetros (Optuna):
  n_neighbors: 0.5500
  metric: 0.2000
  weights: 0.1200
  algorithm: 0.0800
...
"""
    
    # Criar a estrutura: ./results/exp_abril_2025/DECISION_TREE/detailed_results_by_fold.txt
    mock_files = {
        'results/exp_abril_2025/DECISION_TREE/detailed_results_by_fold.txt': file_content_dt,
        'results/exp_abril_2025/K_NEAREST_NEIGHBORS/detailed_results_by_fold.txt': file_content_knn,
        'results/exp_maio_2025/MODELO_X/detailed_results_by_fold.txt': file_content_dt, # Um segundo experimento
    }

    for path, content in mock_files.items():
        # Cria todos os diretórios necessários (ex: results/exp_abril_2025/DECISION_TREE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Escreve o conteúdo do arquivo
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    print("Ambiente de simulação configurado. Execute o script agora.")
# ------------------------------------


def get_experiments(results_dir='./results'):
    """Lista todos os subdiretórios (experimentos) dentro de ./results."""
    if not os.path.isdir(results_dir):
        # Para fins de teste, criamos o ambiente mock se ele não existir
        setup_mock_environment() 
        if not os.path.isdir(results_dir):
             print(f"Erro: O diretório {results_dir} não foi encontrado.")
             return None
             
    experiments = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    return sorted(experiments)

def select_experiment(experiments):
    """Exibe os experimentos e solicita a seleção do usuário."""
    if not experiments:
        print("Nenhum experimento encontrado.")
        return None
        
    print("\n## 🔍 Experimentos Encontrados em './results'")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}: **{exp}**")
        
    while True:
        try:
            selection = input("\nSelecione o número do experimento que deseja analisar: ")
            index = int(selection) - 1
            if 0 <= index < len(experiments):
                selected_exp = experiments[index]
                print(f"**Selecionado:** {selected_exp}")
                return selected_exp
            else:
                print("Seleção inválida. Por favor, digite um número da lista.")
        except ValueError:
            print("Entrada inválida. Por favor, digite apenas o número.")


def extract_importance_from_file(file_path, model_name):
    """
    Extrai a importância dos hiperparâmetros para cada fold de um único arquivo.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  [ERRO] Não foi possível ler o arquivo em {file_path}. Erro: {e}")
        return []

    # Regex para encontrar blocos de importância
    # Procura por "Importância dos parâmetros (Optuna):" seguido de parâmetros
    folds_data = re.findall(
        r'Importância dos parâmetros \(Optuna\):\s*\n((?:\s+[\w_]+:\s*[0-9.]+\s*\n)+)',
        content,
        re.MULTILINE
    )

    importance_data = []
    
    for i, fold_block in enumerate(folds_data):
        fold_importances = {'Modelo': model_name, 'Fold': i + 1}
        # Extrair cada par parâmetro: score
        param_matches = re.findall(r'\s*([\w_]+):\s*([0-9.]+)', fold_block)
        
        for param, score in param_matches:
            fold_importances[param] = float(score)
        
        if fold_importances:
            importance_data.append(fold_importances)

    return importance_data


def consolidate_data(experiment_dir):
    """Percorre as subpastas do modelo e extrai os dados."""
    all_consolidated_data = []
    
    # 1. Encontrar todas as pastas de modelos (subdiretórios)
    model_dirs = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]
    
    if not model_dirs:
        print(f"Aviso: Nenhuma pasta de modelo encontrada em {experiment_dir}.")
        return all_consolidated_data
        
    print(f"\n## 📂 Analisando {len(model_dirs)} modelos em {os.path.basename(experiment_dir)}...")

    for model_name in model_dirs:
        model_path = os.path.join(experiment_dir, model_name)
        file_name = 'detailed_results_by_fold.txt'
        file_path = os.path.join(model_path, file_name)
        
        if os.path.exists(file_path):
            print(f"  - Processando {model_name}...")
            data = extract_importance_from_file(file_path, model_name)
            all_consolidated_data.extend(data)
        else:
            print(f"  - [PULAR] Arquivo {file_name} não encontrado para o modelo {model_name}.")

    return all_consolidated_data


def save_to_excel(experiment_name, data, output_dir="."):
    """Salva os dados consolidados em um arquivo Excel com múltiplas abas (uma por modelo)."""
    if not data:
        print("Nenhum dado consolidado para salvar.")
        return

    output_filename = f"{experiment_name}_importancia_consolidada.xlsx"
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Agrupar dados por modelo
        models_data = {}
        for row in data:
            model_name = row['Modelo']
            if model_name not in models_data:
                models_data[model_name] = []
            models_data[model_name].append(row)
        
        # Criar Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for model_name, model_rows in models_data.items():
                # Converter para DataFrame com importâncias (percentuais)
                df_importance = pd.DataFrame(model_rows)
                
                # Reorganizar colunas para ter Modelo e Fold primeiro
                cols = ['Modelo', 'Fold'] + [col for col in df_importance.columns if col not in ['Modelo', 'Fold']]
                df_importance = df_importance[cols]
                
                # Limitar nome da aba a 31 caracteres (limite do Excel)
                sheet_name = model_name[:31]
                
                # Escrever tabela de importâncias
                df_importance.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        
        print(f"\n✅ Dados consolidados salvos com sucesso em: **{output_path}**")
        print(f"   📊 {len(models_data)} modelos salvos em abas separadas")
    except Exception as e:
        print(f"Erro ao escrever o arquivo Excel: {e}")

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    
    # IMPORTANTE: Descomente a linha abaixo para criar o ambiente de simulação 
    # se você ainda não tem a pasta ./results pronta.
    # setup_mock_environment() 
    
    RESULTS_DIR = './results'
    
    # 1. Listar experimentos
    experiments = get_experiments(RESULTS_DIR)
    if not experiments:
        exit()
        
    # 2. Selecionar experimento
    selected_experiment = select_experiment(experiments)
    if not selected_experiment:
        exit()

    experiment_path = os.path.join(RESULTS_DIR, selected_experiment)
    
    # 3. Consolidar os dados
    consolidated_data = consolidate_data(experiment_path)
    
    # 4. Salvar em Excel (múltiplas abas) dentro da pasta do experimento
    if consolidated_data:
        save_to_excel(selected_experiment, consolidated_data, output_dir=experiment_path)
    else:
        print("\nNão foi possível extrair dados de nenhum modelo. Verifique os arquivos.")