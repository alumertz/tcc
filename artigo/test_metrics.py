#!/usr/bin/env python3

import sys
sys.path.append('/Users/i583975/git/tcc')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from models import evaluate_classification_on_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cria dados sintéticos para teste
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                          n_informative=10, random_state=42)

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina um modelo simples
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)

# Testa a função de avaliação
print("Testando função de avaliação com novas métricas...")
test_metrics = evaluate_classification_on_test(pipeline, X_test, y_test, return_dict=True)

print("\nMétricas retornadas:")
for metric, value in test_metrics.items():
    if metric != 'classification_report':
        print(f"{metric}: {value:.4f}")

print("Teste bem-sucedido!")
