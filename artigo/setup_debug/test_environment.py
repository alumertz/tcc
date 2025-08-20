#!/usr/bin/env python3
"""
Script de teste para verificar se todas as dependências estão instaladas
"""

def test_imports():
    try:
        import pandas as pd
        print("✅ pandas: OK")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy: OK")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn: OK")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import optuna
        print("✅ optuna: OK")
    except ImportError as e:
        print(f"❌ optuna: {e}")
        return False
    
    return True

def test_files():
    import os
    
    features_path = "/Users/i583975/git/tcc/renan/data_files/omics_features/UNION_features.tsv"
    labels_path = "/Users/i583975/git/tcc/renan/data_files/labels/UNION_labels.tsv"
    
    print("\n🔍 Verificando arquivos de dados:")
    
    if os.path.exists(features_path):
        print(f"✅ Features: {features_path}")
    else:
        print(f"❌ Features não encontrado: {features_path}")
        return False
    
    if os.path.exists(labels_path):
        print(f"✅ Labels: {labels_path}")
    else:
        print(f"❌ Labels não encontrado: {labels_path}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔬 TESTE DE CONFIGURAÇÃO DO AMBIENTE")
    print("="*50)
    
    print("\n📦 Testando importações:")
    imports_ok = test_imports()
    
    files_ok = test_files()
    
    print("\n" + "="*50)
    if imports_ok and files_ok:
        print("🎉 AMBIENTE CONFIGURADO CORRETAMENTE!")
        print("✅ Pronto para executar os experimentos")
    else:
        print("❌ PROBLEMAS ENCONTRADOS NO AMBIENTE")
        print("🛠️  Verifique as dependências e arquivos")
