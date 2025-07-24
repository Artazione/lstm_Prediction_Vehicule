#!/usr/bin/env python3
"""
Script: main.py
Auteur: Jules Lefèvre <jules.lefevre@etudiant.univ-reims.fr>
Date de création: 24/07/2025
Description: Point d'entrée unique pour le projet de prédiction de flux véhicules LSTM.
            
            Ce launcher permet d'accéder aux trois composants principaux:
            1. Tuning des hyperparamètres (tuning_lstm.py)
            2. Entraînement et évaluation de modèles (lstm.py)  
            3. Analyse de similarité entre modèles (analyse_similarite_lstm.py)
            
            Usage:
                # Interface de sélection interactive
                python main.py
                
                # Lancement direct d'un composant
                python main.py --component tuning --data ./data --trials 50
                python main.py --component lstm --cli --data ./data --output ./results
                python main.py --component similarity
                
                # Aide détaillée pour chaque composant
                python main.py --help-component tuning
                python main.py --help-component lstm  
                python main.py --help-component similarity
"""

import sys
import os
import argparse
import subprocess
import importlib.util
from pathlib import Path

# =============================================================================
# CONFIGURATION DES COMPOSANTS
# =============================================================================

COMPONENTS = {
    'tuning': {
        'script': 'tuning_lstm.py',
        'description': 'Optimisation d\'hyperparamètres LSTM avec Optuna',
        'requirements': ['optuna', 'torch', 'sklearn'],
        'gui_available': False
    },
    'lstm': {
        'script': 'lstm.py', 
        'description': 'Entraînement et évaluation de modèles LSTM',
        'requirements': ['torch', 'sklearn'],
        'optional_requirements': ['streamlit', 'matplotlib', 'tabulate'],
        'gui_available': True,
        'streamlit_app': True  # Indique que c'est une app Streamlit
    },
    'similarity': {
        'script': 'analyse_similarite_lstm.py',
        'description': 'Analyse de similarité entre modèles LSTM',
        'requirements': ['streamlit', 'torch', 'plotly', 'sklearn'],
        'gui_available': True,
        'streamlit_app': True  # Indique que c'est une app Streamlit
    }
}

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def print_banner():
    """Affiche la bannière du projet."""
    print("=" * 80)
    print("🚦 PROJET PRÉDICTION DE FLUX VÉHICULES - MODÈLES LSTM")
    print("=" * 80)
    print("Auteur: Jules Lefèvre <jules.lefevre@etudiant.univ-reims.fr>")
    print("=" * 80)

def check_dependency(package_name):
    """
    Vérifie si un package Python est disponible.
    
    Args:
        package_name (str): Nom du package à vérifier
        
    Returns:
        bool: True si le package est disponible, False sinon
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_script_exists(script_name):
    """
    Vérifie si un script existe dans le répertoire courant.
    
    Args:
        script_name (str): Nom du script à vérifier
        
    Returns:
        bool: True si le script existe, False sinon
    """
    return Path(script_name).exists()

def get_missing_dependencies(component_name):
    """
    Retourne la liste des dépendances manquantes pour un composant.
    
    Args:
        component_name (str): Nom du composant
        
    Returns:
        tuple: (required_missing, optional_missing)
    """
    component = COMPONENTS[component_name]
    
    required_missing = []
    optional_missing = []
    
    # Vérification des dépendances requises
    for req in component.get('requirements', []):
        if not check_dependency(req):
            required_missing.append(req)
    
    # Vérification des dépendances optionnelles
    for opt in component.get('optional_requirements', []):
        if not check_dependency(opt):
            optional_missing.append(opt)
    
    return required_missing, optional_missing

def print_component_status():
    """Affiche le statut de tous les composants."""
    print("\n📊 STATUT DES COMPOSANTS")
    print("-" * 50)
    
    for comp_name, comp_info in COMPONENTS.items():
        script_exists = check_script_exists(comp_info['script'])
        required_missing, optional_missing = get_missing_dependencies(comp_name)
        
        # Symbole de statut
        if not script_exists:
            status = "❌"
            status_text = "Script manquant"
        elif required_missing:
            status = "⚠️ "
            status_text = f"Dépendances manquantes: {', '.join(required_missing)}"
        elif optional_missing:
            status = "🟡"
            status_text = f"Fonctionnalités limitées (manque: {', '.join(optional_missing)})"
        else:
            status = "✅"
            status_text = "Prêt"
        
        print(f"{status} {comp_name:10} - {comp_info['description']}")
        if status != "✅":
            print(f"{'':13} {status_text}")

def print_installation_help(component_name):
    """Affiche l'aide d'installation pour un composant."""
    required_missing, optional_missing = get_missing_dependencies(component_name)
    
    if required_missing or optional_missing:
        print(f"\n💡 INSTALLATION POUR {component_name.upper()}")
        print("-" * 40)
        
        if required_missing:
            print("Dépendances requises:")
            print(f"  pip install {' '.join(required_missing)}")
        
        if optional_missing:
            print("Dépendances optionnelles (fonctionnalités avancées):")
            print(f"  pip install {' '.join(optional_missing)}")

def select_component_interactive():
    """Interface interactive pour sélectionner un composant."""
    print("\n🎯 SÉLECTION DU COMPOSANT")
    print("-" * 30)
    
    available_components = []
    for i, (comp_name, comp_info) in enumerate(COMPONENTS.items(), 1):
        script_exists = check_script_exists(comp_info['script'])
        required_missing, _ = get_missing_dependencies(comp_name)
        
        if script_exists and not required_missing:
            available_components.append(comp_name)
            print(f"{i}. {comp_name:10} - {comp_info['description']}")
        else:
            print(f"{i}. {comp_name:10} - {comp_info['description']} (❌ Non disponible)")
    
    if not available_components:
        print("\n❌ Aucun composant n'est disponible.")
        print("Vérifiez l'installation des dépendances avec --status")
        return None
    
    print("0. Quitter")
    
    while True:
        try:
            choice = input(f"\nSélectionnez un composant (0-{len(COMPONENTS)}): ").strip()
            
            if choice == '0':
                print("Au revoir !")
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(COMPONENTS):
                comp_name = list(COMPONENTS.keys())[index]
                if comp_name in available_components:
                    return comp_name
                else:
                    print("❌ Ce composant n'est pas disponible.")
                    print_installation_help(comp_name)
            else:
                print(f"⚠️  Veuillez entrer un numéro entre 0 et {len(COMPONENTS)}.")
        
        except ValueError:
            print("❌ Veuillez entrer un numéro valide.")
        except KeyboardInterrupt:
            print("\n\nAu revoir !")
            return None

def run_component(component_name, args):
    """
    Lance un composant avec les arguments fournis.
    
    Args:
        component_name (str): Nom du composant à lancer
        args (list): Arguments à passer au script
    """
    component = COMPONENTS[component_name]
    script_path = component['script']
    
    # Vérifications préalables
    if not check_script_exists(script_path):
        print(f"❌ Script {script_path} introuvable.")
        return False
    
    required_missing, optional_missing = get_missing_dependencies(component_name)
    
    if required_missing:
        print(f"❌ Dépendances manquantes pour {component_name}: {', '.join(required_missing)}")
        print_installation_help(component_name)
        return False
    
    if optional_missing:
        print(f"⚠️  Fonctionnalités limitées (dépendances optionnelles manquantes): {', '.join(optional_missing)}")
    
    # Gestion spéciale pour les applications Streamlit en mode GUI
    if component.get('streamlit_app', False) and not any('--cli' in arg for arg in args):
        # Vérifier que streamlit est disponible
        if not check_dependency('streamlit'):
            print(f"❌ Streamlit requis pour le mode GUI de {component_name}")
            print("Installation: pip install streamlit")
            return False
        
        # Lancement via streamlit run pour le mode GUI
        cmd = ['streamlit', 'run', script_path] + args
        print(f"🚀 Lancement de {component_name} (Interface Streamlit)...")
        print(f"Commande: {' '.join(cmd)}")
        print("📡 L'interface web va s'ouvrir dans votre navigateur...")
    else:
        # Lancement Python standard pour CLI ou scripts non-Streamlit
        cmd = [sys.executable, script_path] + args
        print(f"🚀 Lancement de {component_name}...")
        print(f"Commande: {' '.join(cmd)}")
    
    print("-" * 50)
    
    try:
        # Lancement du script avec transmission des stdin/stdout/stderr
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Interruption par l'utilisateur.")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        return False

def print_component_help(component_name):
    """Affiche l'aide détaillée pour un composant."""
    if component_name not in COMPONENTS:
        print(f"❌ Composant '{component_name}' inconnu.")
        return
    
    component = COMPONENTS[component_name]
    script_path = component['script']
    
    if not check_script_exists(script_path):
        print(f"❌ Script {script_path} introuvable.")
        return
    
    print(f"\n📖 AIDE POUR {component_name.upper()}")
    print("=" * 50)
    
    # Lancement du script avec --help
    cmd = [sys.executable, script_path, '--help']
    subprocess.run(cmd)

# =============================================================================
# FONCTION PRINCIPALE ET PARSING DES ARGUMENTS
# =============================================================================

def main():
    """Point d'entrée principal du launcher."""
    
    parser = argparse.ArgumentParser(
        description="Launcher unique pour le projet de prédiction de flux véhicules LSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Composants disponibles:
  tuning     - Optimisation d'hyperparamètres avec Optuna
  lstm       - Entraînement et évaluation de modèles LSTM
  similarity - Analyse de similarité entre modèles

Exemples d'utilisation:
  
  # Interface interactive
  python main.py
  
  # Lancement direct
  python main.py --component tuning --data ./data --trials 50
  python main.py --component lstm --cli --data ./data --output ./results
  python main.py --component similarity
  python main.py --component similarity --cli --data ./data --models ./models
  
  # Statut et aide
  python main.py --status
  python main.py --help-component lstm
        """
    )
    
    # Arguments principaux
    parser.add_argument(
        '--component', '-c',
        choices=list(COMPONENTS.keys()),
        help='Composant à lancer directement'
    )
    
    parser.add_argument(
        '--status',
        action='store_true', 
        help='Affiche le statut de tous les composants'
    )
    
    parser.add_argument(
        '--help-component',
        choices=list(COMPONENTS.keys()),
        help='Affiche l\'aide détaillée d\'un composant'
    )
    
    # Parsing des arguments connus seulement pour éviter les conflits
    args, remaining_args = parser.parse_known_args()
    
    # Affichage de la bannière
    print_banner()
    
    # Gestion des options spéciales
    if args.status:
        print_component_status()
        
        # Affichage des instructions d'installation si nécessaire
        for comp_name in COMPONENTS:
            print_installation_help(comp_name)
        return
    
    if args.help_component:
        print_component_help(args.help_component)
        return
    
    # Lancement direct d'un composant
    if args.component:
        success = run_component(args.component, remaining_args)
        sys.exit(0 if success else 1)
    
    # Mode interactif si aucun composant spécifié
    print_component_status()
    
    selected_component = select_component_interactive()
    if selected_component:
        
        # Collecte des arguments supplémentaires en mode interactif
        component = COMPONENTS[selected_component]
        
        print(f"\n🔧 CONFIGURATION DE {selected_component.upper()}")
        print("-" * 40)
        
        # Suggestions d'arguments selon le composant
        if selected_component == 'tuning':
            print("Arguments suggérés:")
            print("  --data <dossier>     Dossier des données (requis)")
            print("  --trials <nombre>    Nombre d'essais Optuna (défaut: 100)")
            print("  --threshold <valeur> Seuil MAE% (défaut: 1.0)")
            
        elif selected_component == 'lstm':
            print("Modes disponibles:")
            print("  GUI: (pas d'arguments supplémentaires)")
            print("  CLI: --cli --data <dossier> --output <dossier>")
            print("       Options: --verbose, --quiet")
            
        elif selected_component == 'similarity':
            print("Modes disponibles:")
            print("  GUI: (pas d'arguments supplémentaires - lance Streamlit)")
            print("  CLI: --cli --data <dossier> --models <dossier> --output <dossier>")
            print("       Options: --threshold <seuil>, --verbose, --quiet")
        
        additional_args = input("\nArguments supplémentaires (ou Entrée pour défaut): ").strip()
        
        if additional_args:
            import shlex
            extra_args = shlex.split(additional_args)
        else:
            extra_args = []
        
        success = run_component(selected_component, extra_args)
        sys.exit(0 if success else 1)

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()
