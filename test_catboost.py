#!/usr/bin/env python3
"""
Teste simples para verificar se o CatBoost está funcionando corretamente.
"""

import sys
import os
sys.path.append('/Users/i583975/git/tcc')

import numpy as np
from sklearn.datasets import make_classification

def test_catboost():
    """Teste básico do CatBoost"""
    print("🧪 TESTANDO CATBOOST")
    print("="*40)
    
    try:
        # Importar função
        from src.models import optimize_catboost_classifier
        print("✅ Import do CatBoost OK")
        
        # Criar dataset de teste pequeno
        X, y = make_classification(
            n_samples=100, 
            n_features=10, 
            n_classes=2, 
            random_state=42
        )
        print(f"✅ Dataset criado: {X.shape}")
        
        # Testar com poucos trials
        print("🔄 Executando otimização (3 trials)...")
        model = optimize_catboost_classifier(
            X, y, 
            n_trials=3, 
            save_results=False
        )
        
        if model is not None:
            print("✅ CatBoost funcionando!")
            print(f"✅ Tipo do modelo: {type(model)}")
            
            # Testar predição
            predictions = model.predict(X[:5])
            print(f"✅ Predições: {predictions}")
            
            return True
        else:
            print("❌ Modelo retornou None")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_catboost()
    
    if success:
        print("\n🎉 CATBOOST FUNCIONANDO CORRETAMENTE!")
        print("Pode ser usado no pipeline principal.")
    else:
        print("\n❌ CATBOOST COM PROBLEMAS!")
        print("Verificar configuração.")
