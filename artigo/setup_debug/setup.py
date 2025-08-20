#!/usr/bin/env python3
"""
Script de inicialização do projeto de classificação de genes-alvo.
Verifica e instala dependências automaticamente.
"""

import subprocess
import sys
import os

def check_python_version():
    """Verifica se a versão do Python é adequada"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_package(package):
    """Instala um pacote usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Verifica e instala dependências necessárias"""
    required_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "optuna"
    ]
    
    print("\n📦 Verificando dependências...")
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - será instalado")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 Instalando {len(missing_packages)} pacote(s)...")
        
        for package in missing_packages:
            print(f"   Instalando {package}...")
            if install_package(package):
                print(f"   ✅ {package} instalado")
            else:
                print(f"   ❌ Erro ao instalar {package}")
                return False
    
    return True

def check_data_files():
    """Verifica se os arquivos de dados existem"""
    print("\n📁 Verificando arquivos de dados...")
    
    base_path = "/Users/i583975/git/tcc/renan/data_files"
    
    files_to_check = [
        ("Features", f"{base_path}/omics_features/UNION_features.tsv"),
        ("Labels", f"{base_path}/labels/UNION_labels.tsv")
    ]
    
    all_found = True
    
    for name, path in files_to_check:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ {name}: {path} ({size:,} bytes)")
        else:
            print(f"❌ {name}: {path} - NÃO ENCONTRADO")
            all_found = False
    
    return all_found

def check_models_file():
    """Verifica se o arquivo models.py existe e não está vazio"""
    models_path = "/Users/i583975/git/tcc/models.py"
    
    print("\n🤖 Verificando arquivo de modelos...")
    
    if not os.path.exists(models_path):
        print(f"❌ models.py não encontrado: {models_path}")
        return False
    
    size = os.path.getsize(models_path)
    if size == 0:
        print(f"❌ models.py está vazio: {models_path}")
        return False
    
    print(f"✅ models.py: {models_path} ({size:,} bytes)")
    return True

def setup_project():
    """Configuração completa do projeto"""
    print("🧬 CONFIGURAÇÃO DO PROJETO - CLASSIFICAÇÃO DE GENES-ALVO")
    print("="*80)
    
    # Verificar versão do Python
    if not check_python_version():
        return False
    
    # Verificar e instalar dependências
    if not check_and_install_dependencies():
        print("\n❌ Erro ao instalar dependências")
        return False
    
    # Verificar arquivos de dados
    if not check_data_files():
        print("\n❌ Arquivos de dados não encontrados")
        print("   Verifique se o diretório renan/data_files está correto")
        return False
    
    # Verificar arquivo de modelos
    if not check_models_file():
        print("\n❌ Arquivo models.py com problemas")
        return False
    
    print("\n" + "="*80)
    print("🎉 PROJETO CONFIGURADO COM SUCESSO!")
    print("\n📋 Próximos passos:")
    print("   1. Teste o ambiente: python test_environment.py")
    print("   2. Execute exemplo: python exemplo.py")
    print("   3. Experimento completo: python main.py")
    print("="*80)
    
    return True

def main():
    """Função principal"""
    success = setup_project()
    
    if not success:
        print("\n💡 Para resolver problemas:")
        print("   - Verifique se está no diretório correto")
        print("   - Ative o ambiente virtual se necessário")
        print("   - Execute novamente este script")
        sys.exit(1)

if __name__ == "__main__":
    main()
