# 🧬 Classificação de Genes-Alvo usando Dados Multi-Ômicos

Este projeto implementa uma pipeline completa para classificação de genes-alvo usando dados multi-ômicos (CNA, Gene Expression, Methylation, Mutation Frequency) com otimização automática de hiperparâmetros usando Optuna.

## 📁 Estrutura do Projeto

```
/tcc/artigo/
├── 📋 README.md              # Documentação completa do projeto
├── 🚀 main.py               # Arquivo principal para experimentos
├── ⚙️  processamento.py      # Funções de processamento de dados
├── 🧪 exemplo.py            # Exemplos de uso individual
├── 🔧 setup.py              # Script de configuração automática
├── 🔍 test_environment.py   # Teste de ambiente e dependências
└── 📁 results/              # Diretório para resultados organizados
```

## 📦 Modelos Implementados

✅ **7 modelos de classificação com otimização Optuna:**

1. 🌳 **Decision Tree Classifier** - Árvore de decisão com otimização de profundidade e critério
2. 🌲 **Random Forest Classifier** - Ensemble de árvores com otimização de estimadores e features
3. 📈 **Gradient Boosting Classifier** - Boosting gradiente com otimização de learning rate
4. 📊 **Histogram Gradient Boosting Classifier** - Versão otimizada do gradient boosting
5. 🎯 **K-Nearest Neighbors Classifier** - KNN com otimização de vizinhos e métricas de distância
6. 🧠 **Multi-Layer Perceptron Classifier** - Rede neural com arquitetura flexível
7. ⚡ **Support Vector Classifier** - SVM com diferentes kernels

## 🛠️ Dependências

- **Python 3.8+**
- **pandas**: Manipulação de dados
- **numpy**: Operações numéricas  
- **scikit-learn**: Modelos de ML
- **optuna**: Otimização de hiperparâmetros
- **imblearn**: Balanceamento de classes (SMOTE)

## 🚀 Instalação e Configuração

### 1️⃣ Configuração Inicial
```bash
cd /Users/i583975/git/tcc/artigo
python setup.py
```

**Ou manualmente:**

1. **Criar ambiente virtual:**
```bash
cd /Users/i583975/git/tcc
python3 -m venv mlenv
source mlenv/bin/activate
```

2. **Instalar dependências:**
```bash
pip install pandas numpy scikit-learn optuna imbalanced-learn
```

## 📋 Dados Necessários

- 📄 **UNION_features.tsv**: Features multi-ômicas (CNA, GE, METH, MF)
- 🏷️ **UNION_labels.tsv**: Labels de classificação (True/False)

**Localização:**
- Features: `/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv`
- Labels: `/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv`

## 🚀 Como Usar

### 1️⃣ Teste do Ambiente

```bash
cd /Users/i583975/git/tcc/artigo
python test_environment.py
```

### 2️⃣ Exemplo Individual

```bash
python exemplo.py
```

### 3️⃣ Experimento Completo

```bash
python main.py
```

### 4️⃣ Uso Programático

```python
import sys
sys.path.append('/Users/i583975/git/tcc')

from processamento import prepare_dataset
from models import optimize_random_forest_classifier

# Preparar dados
X, y, gene_names, feature_names = prepare_dataset(
    "/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv",
    "/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv"
)

# Executar Random Forest
best_model = optimize_random_forest_classifier(X, y, n_trials=50)
```

## 🔄 Funcionalidades

### 📊 Processamento de Dados
- ✅ Carregamento automático de `UNION_features.tsv` e `UNION_labels.tsv`
- ✅ Alinhamento automático de features e labels por gene
- ✅ Tratamento de valores faltantes (preenchimento com 0)
- ✅ Conversão automática de labels (True/False → 1/0)
- ✅ Análise exploratória automática do dataset

### 🎛️ Otimização de Hiperparâmetros
- ✅ **Optuna**: Framework para otimização bayesiana
- ✅ **Validação cruzada estratificada**: 5-fold (adaptativo para datasets pequenos)
- ✅ **Holdout 80/20**: Divisão para treino/validação e teste final
- ✅ **Configuração flexível**: Número de trials personalizável

### 📈 Avaliação Completa
- ✅ **Métricas**: Acurácia, Precisão, Recall, F1-Score
- ✅ **Classification Report**: Relatório detalhado por classe
- ✅ **Pipelines**: Integração automática com StandardScaler
- ✅ **Tracking**: Tempo de execução por trial e modelo

### 💾 Sistema de Resultados
- ✅ **Salvamento automático** de resultados em arquivos organizados
- ✅ **Estrutura por modelo**: Diretórios separados para cada algoritmo
- ✅ **Formatos múltiplos**: JSON (trials) + TXT (relatórios)
- ✅ **Timestamps**: Arquivos com data/hora para versionamento

## ⚙️ Características da Implementação

### 🎯 Validação Robusta
- **Estratificação**: Mantém proporção das classes em todas as divisões
- **Holdout 80/20**: Divisão inicial para treino/validação e teste
- **5-Fold Stratified CV**: Validação cruzada estratificada interna (adaptativo)
- **Cross-Validation**: Ajusta automaticamente o número de folds para datasets pequenos

### 🔧 Otimização Automática
- **Optuna**: Framework para otimização bayesiana de hiperparâmetros
- **Trials configuráveis**: Número de tentativas por modelo (padrão: 30)
- **Métricas**: Acurácia como métrica principal de otimização
- **Sampling**: TPE (Tree-structured Parzen Estimator) para eficiência

### 📊 Avaliação Completa
- **Acurácia**: Métrica principal para comparação
- **Precisão**: Média ponderada por classe
- **Recall**: Média ponderada por classe  
- **F1-Score**: Média ponderada por classe
- **Classification Report**: Relatório detalhado por classe

### 🔄 Pré-processamento
- **StandardScaler**: Normalização automática das features
- **Pipeline**: Integração transparente de preprocessamento e modelo
- **SMOTE**: Técnica de oversampling para balanceamento (quando necessário)

## 🔢 Configuração dos Hiperparâmetros

Cada modelo tem ranges específicos de hiperparâmetros otimizados:

### 🌳 Decision Tree
- `max_depth`: 2-32
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-20
- `criterion`: gini, entropy

### 🌲 Random Forest
- `n_estimators`: 100-300 (step 50)
- `max_depth`: 5-30
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-20
- `max_features`: sqrt, log2, None
- `criterion`: gini, entropy

### 📈 Gradient Boosting
- `n_estimators`: 100-300 (step 50)
- `learning_rate`: 0.01-0.3
- `max_depth`: 3-10
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-20
- `subsample`: 0.8-1.0

### 📊 Histogram Gradient Boosting
- `max_iter`: 50-200
- `learning_rate`: 0.01-0.3
- `max_depth`: 3-15
- `min_samples_leaf`: 1-50
- `l2_regularization`: 0.0-1.0

### 🎯 K-Nearest Neighbors
- `n_neighbors`: 1-20
- `weights`: uniform, distance
- `algorithm`: auto, ball_tree, kd_tree, brute
- `p`: 1 (manhattan), 2 (euclidean)

### 🧠 Multi-Layer Perceptron
- `n_layers`: 1-3
- `layer_size`: 10-200 (por camada)
- `activation`: tanh, relu, logistic
- `alpha`: 1e-5 to 1e-1 (log scale)
- `learning_rate`: constant, invscaling, adaptive
- `max_iter`: 200-1000

### ⚡ Support Vector Classifier
- `kernel`: linear, poly, rbf, sigmoid
- `C`: 1e-3 to 1e3 (log scale)
- `gamma`: scale, auto (para kernels não-lineares)
- `degree`: 2-5 (apenas para kernel poly)

## 📈 Output Esperado

O experimento produz:

### 📊 Logs Detalhados
- Progresso de cada trial de otimização
- Melhores hiperparâmetros encontrados para cada modelo
- Tempo de execução por trial e total
- Avaliação completa no conjunto de teste

### 💾 Arquivos de Resultados
- **`/results/{modelo}/trials_{timestamp}.json`**: Histórico completo de trials
- **`/results/{modelo}/test_results_{timestamp}.txt`**: Relatório formatado dos resultados

### 📋 Resumo Final
- Modelos bem-sucedidos vs com erro
- Comparação de performance entre algoritmos
- Distribuição final das classes no dataset

## 💡 Exemplos de Uso

### 🔬 Modelo Individual
```python
from processamento import prepare_dataset
from models import optimize_random_forest_classifier

# Carregar dados
X, y, genes, features = prepare_dataset(features_path, labels_path)

# Otimizar Random Forest
best_model = optimize_random_forest_classifier(X, y, n_trials=50)
```

### 📊 Comparação de Modelos
```python
# Ver exemplo completo em exemplo.py
python exemplo.py
```

### 🚀 Experimento Completo
```python
# Ver configuração completa em main.py
python main.py
```

### 📁 Análise de Resultados
```python
import json
import pandas as pd

# Carregar resultados de trials
with open('/results/random_forest/trials_20250820_123456.json', 'r') as f:
    trials_data = json.load(f)

# Converter para DataFrame para análise
trials_df = pd.DataFrame(trials_data)
print(f"Melhor trial: {trials_df.loc[trials_df['score'].idxmax()]}")
```

## 🔧 Personalização

### 🎛️ Alterar Número de Trials
```python
# Edite a variável N_TRIALS no arquivo main.py (padrão: 30)
N_TRIALS = 50  # Para otimização mais demorada mas possivelmente melhor
```

### 📊 Modificar Métrica de Otimização
```python
# Altere o parâmetro scoring nas funções de otimização
score = cross_val_score(
    pipeline, X_trainval, y_trainval,
    cv=inner_cv,
    scoring="f1_weighted"  # Padrão: "accuracy"
).mean()
```

### ➕ Adicionar Novos Modelos
1. Implemente nova função de otimização em `/Users/i583975/git/tcc/models.py`
2. Siga o padrão: `optimize_{nome}_classifier(X, y, n_trials=30, save_results=True)`
3. Adicione ao `models_config` em `main.py`

### 🎯 Balanceamento de Classes
```python
# Para datasets muito desbalanceados, considere usar SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

## 🎯 Próximos Passos

1. ✅ **Execute `setup.py`** para verificar ambiente
2. ✅ **Teste com `test_environment.py`** para validar instalação
3. ✅ **Execute `exemplo.py`** para validar funcionamento
4. ✅ **Execute `main.py`** para experimento completo
5. 📊 **Analise resultados** na pasta `/results/`
6. 💾 **Salve melhores modelos** para uso futuro

## 🚨 Troubleshooting

### ❌ Erro de Módulo Não Encontrado
```bash
# Ativar ambiente virtual
source /Users/i583975/git/tcc/mlenv/bin/activate

# Instalar dependências faltantes
pip install pandas scikit-learn optuna imbalanced-learn
```

### 📁 Arquivo de Dados Não Encontrado
Verifique se os caminhos dos arquivos estão corretos:
- `UNION_features.tsv`: Features multi-ômicas
- `UNION_labels.tsv`: Labels de classificação

### 💾 Problemas de Memória
- Reduza o número de trials (`N_TRIALS = 10`)
- Use menos folds na validação cruzada (modifique `get_optimal_cv_folds`)
- Filtre features menos relevantes antes do treinamento

### ⚠️ Dataset Desbalanceado
- O projeto já trata automaticamente datasets desbalanceados
- Classes minoritárias são preservadas na estratificação
- Considere métricas como F1-Score para datasets muito desbalanceados

### 🐛 Erros de Convergência (MLP/SVC)
- Aumente `max_iter` para MLP
- Reduza o range de `C` para SVC
- Use features normalizadas (já implementado via Pipeline)

## 💾 Resultados e Persistência

### 🔄 Salvamento Automático
Os modelos treinados podem ser salvos e reutilizados:

```python
import joblib

# Salvar modelo após otimização
joblib.dump(best_model, 'models/best_random_forest_model.pkl')

# Carregar modelo salvo
loaded_model = joblib.load('models/best_random_forest_model.pkl')

# Fazer predições em novos dados
predictions = loaded_model.predict(new_data)
probabilities = loaded_model.predict_proba(new_data)
```

### 📊 Análise de Performance
```python
# Comparar resultados de múltiplos modelos
import os
import json

results_dir = '/Users/i583975/git/tcc/artigo/results'
models_performance = {}

for model_name in os.listdir(results_dir):
    model_dir = os.path.join(results_dir, model_name)
    if os.path.isdir(model_dir):
        # Encontrar arquivo de resultados mais recente
        result_files = [f for f in os.listdir(model_dir) if f.startswith('test_results_')]
        if result_files:
            latest_file = sorted(result_files)[-1]
            # Extrair métricas do arquivo
            models_performance[model_name] = latest_file

print("Performance dos modelos:", models_performance)
```

### 📈 Visualização de Resultados
```python
import matplotlib.pyplot as plt
import pandas as pd

# Exemplo de visualização de trials do Optuna
with open('results/random_forest/trials_latest.json', 'r') as f:
    trials = json.load(f)

df_trials = pd.DataFrame(trials)
plt.figure(figsize=(10, 6))
plt.plot(df_trials['trial_number'], df_trials['score'])
plt.title('Evolução da Performance durante Otimização')
plt.xlabel('Trial Number')
plt.ylabel('Accuracy Score')
plt.show()
```

---

## 🎉 Status do Projeto

**✅ PROJETO PRONTO PARA USO!**

Todos os arquivos foram criados e estão funcionais. O sistema está configurado para classificação de genes-alvo usando dados multi-ômicos com otimização automática de hiperparâmetros.

### 📋 Checklist de Funcionalidades

- [x] **Processamento de dados** multi-ômicos
- [x] **7 algoritmos de ML** implementados
- [x] **Otimização automática** com Optuna
- [x] **Validação robusta** com CV estratificada
- [x] **Sistema de resultados** organizados
- [x] **Documentação completa** e exemplos
- [x] **Tratamento de erros** e edge cases
- [x] **Pipeline automatizada** end-to-end

### 🚀 Capacidades do Sistema

- **Dataset suportado**: 13.825 genes com features multi-ômicas
- **Classes balanceadas**: Tratamento automático de desbalanceamento
- **Escalabilidade**: Configurações adaptáveis ao tamanho do dataset
- **Reprodutibilidade**: Seeds fixadas para resultados consistentes
- **Monitoramento**: Logs detalhados e tracking de performance

---

**Desenvolvido para classificação de oncogenes usando dados multi-ômicos integrados** 🧬
