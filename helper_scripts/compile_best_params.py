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


def get_subdirectories(directory):
    """Lista todos os subdiretórios dentro de um diretório."""
    if not os.path.isdir(directory):
        return []
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return sorted(subdirs)

def is_experiment_folder(folder_path):
    """
    Verifica se uma pasta parece ser um experimento (contém subpastas de modelos).
    Um experimento deve ter pelo menos uma subpasta com o arquivo detailed_results_by_fold.txt
    """
    subdirs = get_subdirectories(folder_path)
    for subdir in subdirs:
        model_path = os.path.join(folder_path, subdir)
        file_path = os.path.join(model_path, 'detailed_results_by_fold.txt')
        if os.path.exists(file_path):
            return True
    return False

def navigate_and_select(current_dir='./results', allow_folder_selection=False):
    """
    Navega recursivamente pelos diretórios e permite selecionar um experimento ou pasta.
    
    Args:
        current_dir: Diretório inicial
        allow_folder_selection: Se True, permite selecionar uma pasta (não apenas experimentos)
    """
    if not os.path.isdir(current_dir):
        # Para fins de teste, criamos o ambiente mock se ele não existir
        setup_mock_environment() 
        if not os.path.isdir(current_dir):
             print(f"Erro: O diretório {current_dir} não foi encontrado.")
             return None
    
    while True:
        subdirs = get_subdirectories(current_dir)
        
        if not subdirs:
            print(f"\nNenhuma subpasta encontrada em {current_dir}.")
            # Se permite seleção de pasta e estamos numa pasta válida, retornar ela
            if allow_folder_selection and current_dir != './results':
                return current_dir
            return None
        
        # Identificar quais são experimentos e quais são apenas pastas
        experiments = []
        folders = []
        
        for d in subdirs:
            full_path = os.path.join(current_dir, d)
            if is_experiment_folder(full_path):
                experiments.append(d)
            else:
                folders.append(d)
        
        print(f"\n## 📂 Navegando em: {current_dir}")
        print("=" * 80)
        
        options = []
        
        # Mostrar opção de voltar se não estiver no diretório raiz
        if current_dir != './results':
            print(f"  0: 🔙 [Voltar para pasta anterior]")
            options.append(('back', None))
        
        # Se permite seleção de pasta E há experimentos aqui, mostrar opção de selecionar esta pasta
        if allow_folder_selection and experiments:
            idx = len(options) + 1
            print(f"  {idx}: 📦 [Selecionar ESTA PASTA para agregação]")
            options.append(('select_current', None))
        
        # Mostrar pastas (não são experimentos)
        if folders:
            print("\n  📁 PASTAS:")
            for folder in folders:
                idx = len(options) + 1
                print(f"  {idx}: 📁 {folder}")
                options.append(('folder', folder))
        
        # Mostrar experimentos
        if experiments:
            print("\n  🔬 EXPERIMENTOS:")
            for exp in experiments:
                idx = len(options) + 1
                print(f"  {idx}: ✨ {exp}")
                options.append(('experiment', exp))
        
        if not options:
            print("\nNenhuma opção disponível.")
            return None
        
        # Solicitar seleção
        while True:
            try:
                selection = input(f"\nSelecione uma opção (0-{len(options)}): ")
                index = int(selection)
                
                if index == 0 and current_dir != './results':
                    # Voltar para pasta anterior
                    current_dir = os.path.dirname(current_dir)
                    if not current_dir or current_dir == '.':
                        current_dir = './results'
                    break
                elif 1 <= index <= len(options):
                    option_type, name = options[index - 1]
                    
                    if option_type == 'select_current':
                        print(f"**Pasta selecionada:** {current_dir}")
                        return current_dir
                    elif option_type == 'experiment':
                        print(f"**Selecionado:** {name}")
                        return os.path.join(current_dir, name)
                    elif option_type == 'folder':
                        # Navegar para dentro da pasta
                        current_dir = os.path.join(current_dir, name)
                        break
                    elif option_type == 'back':
                        # Voltar
                        current_dir = os.path.dirname(current_dir)
                        if not current_dir or current_dir == '.':
                            current_dir = './results'
                        break
                else:
                    print("Seleção inválida. Por favor, digite um número da lista.")
            except ValueError:
                print("Entrada inválida. Por favor, digite apenas o número.")
            except KeyboardInterrupt:
                print("\nOperação cancelada pelo usuário.")
                return None


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


def consolidate_data(experiment_path):
    """Percorre as subpastas do modelo e extrai os dados."""
    all_consolidated_data = []
    
    # 1. Encontrar todas as pastas de modelos (subdiretórios)
    model_dirs = [d for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    
    if not model_dirs:
        print(f"Aviso: Nenhuma pasta de modelo encontrada em {experiment_path}.")
        return all_consolidated_data
        
    print(f"\n## 📂 Analisando {len(model_dirs)} modelos em {os.path.basename(experiment_path)}...")

    for model_name in model_dirs:
        model_path = os.path.join(experiment_path, model_name)
        file_name = 'detailed_results_by_fold.txt'
        file_path = os.path.join(model_path, file_name)
        
        if os.path.exists(file_path):
            print(f"  - Processando {model_name}...")
            data = extract_importance_from_file(file_path, model_name)
            all_consolidated_data.extend(data)
        else:
            print(f"  - [PULAR] Arquivo {file_name} não encontrado para o modelo {model_name}.")

    return all_consolidated_data


def save_to_excel(experiment_path, data, output_dir=None):
    """Salva os dados consolidados em um arquivo Excel com múltiplas abas (uma por modelo)."""
    if not data:
        print("Nenhum dado consolidado para salvar.")
        return

    # Usar o nome da pasta do experimento para o arquivo
    experiment_name = os.path.basename(experiment_path)
    output_filename = f"{experiment_name}_importancia_consolidada.xlsx"
    
    # Se output_dir não for especificado, salvar na pasta do experimento
    if output_dir is None:
        output_dir = experiment_path
    
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
                
                # Calcular média para cada parâmetro (excluindo colunas Modelo e Fold)
                param_cols = [col for col in df_importance.columns if col not in ['Modelo', 'Fold']]
                avg_row = {'Modelo': 'MÉDIA', 'Fold': ''}
                for col in param_cols:
                    avg_row[col] = df_importance[col].mean()
                
                # Adicionar linha de média ao DataFrame
                df_with_avg = pd.concat([df_importance, pd.DataFrame([avg_row])], ignore_index=True)
                
                # Limitar nome da aba a 31 caracteres (limite do Excel)
                sheet_name = model_name[:31]
                
                # Escrever tabela de importâncias com média
                df_with_avg.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        
        print(f"\n✅ Dados consolidados salvos com sucesso em: **{output_path}**")
        print(f"   📊 {len(models_data)} modelos salvos em abas separadas (com linha de média)")
    except Exception as e:
        print(f"Erro ao escrever o arquivo Excel: {e}")


def find_all_experiments_in_folder(parent_folder):
    """
    Encontra todos os experimentos dentro de uma pasta (e subpastas).
    Retorna uma lista de tuplas (nome_experimento, caminho_completo).
    """
    experiments = []
    
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            if is_experiment_folder(item_path):
                experiments.append((item, item_path))
            else:
                # Buscar recursivamente em subpastas
                sub_experiments = find_all_experiments_in_folder(item_path)
                experiments.extend(sub_experiments)
    
    return experiments


def load_experiment_data_from_excel(experiment_path):
    """
    Carrega dados de um experimento a partir do Excel já gerado.
    Retorna um dicionário {modelo: DataFrame}.
    """
    experiment_name = os.path.basename(experiment_path)
    excel_filename = f"{experiment_name}_importancia_consolidada.xlsx"
    excel_path = os.path.join(experiment_path, excel_filename)
    
    if not os.path.exists(excel_path):
        return None
    
    try:
        excel_file = pd.ExcelFile(excel_path)
        models_data = {}
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # Remover linha de média se existir
            df = df[df['Modelo'] != 'MÉDIA']
            models_data[sheet_name] = df
        
        return models_data
    except Exception as e:
        print(f"  [ERRO] Não foi possível ler o Excel em {excel_path}. Erro: {e}")
        return None


def get_experiment_data(experiment_path):
    """
    Obtém dados de um experimento. Tenta primeiro carregar do Excel,
    senão extrai dos arquivos de texto.
    """
    experiment_name = os.path.basename(experiment_path)
    
    # Tentar carregar do Excel primeiro
    excel_data = load_experiment_data_from_excel(experiment_path)
    if excel_data:
        return excel_data
    
    # Se não existe Excel, extrair dos arquivos
    print(f"  - Excel não encontrado para {experiment_name}, extraindo dos arquivos...")
    raw_data = consolidate_data(experiment_path)
    
    if not raw_data:
        return None
    
    # Agrupar por modelo
    models_data = {}
    for row in raw_data:
        model_name = row['Modelo']
        if model_name not in models_data:
            models_data[model_name] = []
        models_data[model_name].append(row)
    
    # Converter para DataFrames
    for model_name in models_data:
        df = pd.DataFrame(models_data[model_name])
        cols = ['Modelo', 'Fold'] + [col for col in df.columns if col not in ['Modelo', 'Fold']]
        df = df[cols]
        models_data[model_name] = df
    
    return models_data


def aggregate_experiments(parent_folder):
    """
    Agrega dados de múltiplos experimentos em um único Excel.
    Cada sheet = um modelo, com múltiplas tabelas (uma por experimento).
    """
    # Encontrar todos os experimentos
    experiments = find_all_experiments_in_folder(parent_folder)
    
    if not experiments:
        print(f"\nNenhum experimento encontrado em {parent_folder}")
        return
    
    print(f"\n## 🔬 Encontrados {len(experiments)} experimentos:")
    for exp_name, exp_path in experiments:
        print(f"  - {exp_name}")
    
    # Coletar dados de todos os experimentos
    all_experiments_data = {}  # {exp_name: {model_name: DataFrame}}
    
    print(f"\n## 📊 Carregando dados dos experimentos...")
    for exp_name, exp_path in experiments:
        print(f"  - Carregando {exp_name}...")
        exp_data = get_experiment_data(exp_path)
        if exp_data:
            all_experiments_data[exp_name] = exp_data
    
    if not all_experiments_data:
        print("\nNenhum dado foi carregado dos experimentos.")
        return
    
    # Identificar todos os modelos únicos
    all_models = set()
    for exp_data in all_experiments_data.values():
        all_models.update(exp_data.keys())
    
    all_models = sorted(all_models)
    
    print(f"\n## 📋 Modelos encontrados: {len(all_models)}")
    for model in all_models:
        print(f"  - {model}")
    
    # Para cada modelo, identificar todos os parâmetros únicos e sua ordem consistente
    model_params = {}  # {model_name: [ordered_param_list]}
    
    for model_name in all_models:
        all_params = set()
        for exp_data in all_experiments_data.values():
            if model_name in exp_data:
                df = exp_data[model_name]
                params = [col for col in df.columns if col not in ['Modelo', 'Fold']]
                all_params.update(params)
        model_params[model_name] = sorted(all_params)
    
    # Criar Excel com múltiplas tabelas por sheet
    parent_folder_name = os.path.basename(parent_folder.rstrip('/'))
    output_filename = f"{parent_folder_name}_AGREGADO.xlsx"
    output_path = os.path.join(parent_folder, output_filename)
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for model_name in all_models:
                current_row = 0
                param_order = model_params[model_name]
                
                for exp_name in sorted(all_experiments_data.keys()):
                    exp_data = all_experiments_data[exp_name]
                    
                    if model_name not in exp_data:
                        continue
                    
                    df = exp_data[model_name].copy()
                    
                    # Garantir que todas as colunas de parâmetros existam (preencher com NaN se ausente)
                    for param in param_order:
                        if param not in df.columns:
                            df[param] = float('nan')
                    
                    # Reorganizar colunas na ordem consistente
                    ordered_cols = ['Modelo', 'Fold'] + param_order
                    df = df[ordered_cols]
                    
                    # Calcular média para cada parâmetro
                    avg_row = {'Modelo': 'MÉDIA', 'Fold': ''}
                    for col in param_order:
                        avg_row[col] = df[col].mean()
                    
                    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                    
                    # Escrever título do experimento
                    sheet_name = model_name[:31]
                    
                    # Criar DataFrame do título
                    title_df = pd.DataFrame([[f"EXPERIMENTO: {exp_name}"]], columns=[''])
                    title_df.to_excel(writer, sheet_name=sheet_name, index=False, 
                                    header=False, startrow=current_row)
                    current_row += 2  # Título + linha em branco
                    
                    # Escrever tabela do experimento
                    df_with_avg.to_excel(writer, sheet_name=sheet_name, index=False, 
                                        startrow=current_row)
                    current_row += len(df_with_avg) + 3  # Dados + header + 2 linhas em branco
        
        print(f"\n✅ Arquivo agregado salvo com sucesso em: **{output_path}**")
        print(f"   📊 {len(all_models)} modelos com dados de {len(all_experiments_data)} experimentos")
    except Exception as e:
        print(f"Erro ao escrever o arquivo Excel agregado: {e}")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    
    # IMPORTANTE: Descomente a linha abaixo para criar o ambiente de simulação 
    # se você ainda não tem a pasta ./results pronta.
    # setup_mock_environment() 
    
    RESULTS_DIR = './results'
    
    # Escolher modo de operação
    print("\n" + "="*80)
    print("  COMPILADOR DE IMPORTÂNCIA DE PARÂMETROS")
    print("="*80)
    print("\nEscolha o modo de operação:")
    print("  1: 📄 Processar um EXPERIMENTO individual")
    print("  2: 📚 AGREGAR múltiplos experimentos de uma pasta")
    
    while True:
        try:
            mode = input("\nSelecione o modo (1 ou 2): ").strip()
            if mode in ['1', '2']:
                break
            print("Opção inválida. Digite 1 ou 2.")
        except KeyboardInterrupt:
            print("\nOperação cancelada.")
            exit()
    
    if mode == '1':
        # MODO 1: Processar experimento individual
        print("\n" + "="*80)
        print("  MODO: Experimento Individual")
        print("="*80)
        
        # 1. Navegar e selecionar experimento
        experiment_path = navigate_and_select(RESULTS_DIR)
        if not experiment_path:
            print("\nNenhum experimento selecionado. Encerrando.")
            exit()
        
        # 2. Consolidar os dados
        consolidated_data = consolidate_data(experiment_path)
        
        # 3. Salvar em Excel (múltiplas abas) dentro da pasta do experimento
        if consolidated_data:
            save_to_excel(experiment_path, consolidated_data)
        else:
            print("\nNão foi possível extrair dados de nenhum modelo. Verifique os arquivos.")
    
    else:
        # MODO 2: Agregar múltiplos experimentos
        print("\n" + "="*80)
        print("  MODO: Agregação de Experimentos")
        print("="*80)
        print("\nSelecione a PASTA que contém os experimentos a agregar:")
        
        # Navegar até a pasta que contém os experimentos (permite seleção de pastas)
        parent_folder = navigate_and_select(RESULTS_DIR, allow_folder_selection=True)
        if not parent_folder:
            print("\nNenhuma pasta selecionada. Encerrando.")
            exit()
        
        # Agregar experimentos
        aggregate_experiments(parent_folder)
    
    print("\n" + "="*80)
    print("  PROCESSAMENTO CONCLUÍDO")
    print("="*80)