"""
Script: lstm.py
Auteur: Jules Lefèvre <jules.lefevre@etudiant.univ-reims.fr>
Date de création: 02/07/2025
Description: Application Streamlit/CLI interactive pour la prédiction de flux de véhicules 
            avec des modèles LSTM. L'application propose trois modes d'utilisation :
            1. Entraînement de nouveaux modèles avec validation croisée optionnelle
            2. Chargement et évaluation de modèles pré-entraînés
            3. Comparaison de performances entre plusieurs modèles
            
            Usage:
            # Mode GUI (Streamlit)
            python lstm.py
            
            # Mode CLI
            python lstm.py --cli --data ./data --output ./resultats
            
            Fonctionnalités principales:
            - Interface utilisateur intuitive avec upload de fichiers CSV (GUI) ou sélection interactive (CLI)
            - Feature engineering automatique (moyennes mobiles, encodage cyclique)
            - Visualisations interactives (GUI) ou sauvegardées (CLI)
            - Sauvegarde automatique des checkpoints avec métadonnées complètes
            - Support multi-capteurs avec traitement parallèle
"""

# =============================================================================
# IMPORTS ET CONFIGURATION DE BASE
# =============================================================================

import argparse
import sys
import os
import glob
import random
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dateutil import parser as date_parser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Imports pour la CLI
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Détection du mode CLI avant d'importer Streamlit
_is_cli_mode = '--cli' in sys.argv

# Import matplotlib avec backend approprié
import matplotlib
if _is_cli_mode:
    matplotlib.use('Agg')  # Backend non-interactif pour CLI
import matplotlib.pyplot as plt

# Imports conditionnels pour Streamlit (seulement si mode GUI)
STREAMLIT_AVAILABLE = False
st = None  # Placeholder pour éviter les erreurs de référence
if not _is_cli_mode:
    try:
        import streamlit as st
        STREAMLIT_AVAILABLE = True
    except ImportError:
        pass

# =============================================================================
# CONFIGURATION GLOBALE ET REPRODUCTIBILITÉ
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"

CLI_MODE = _is_cli_mode
VERBOSE_LEVEL = 1  # 0: quiet, 1: normal, 2: verbose

# Features globales
FEATURE_COLS_LSTM = ['hour_cos', 'mean_flow_others', 'ma3', 'ma6', 'ma12']
FEATURE_COLS_TUNING = ['flow[veh/h]', 'hour_cos', 'mean_flow_others', 'ma3', 'ma6', 'ma12']
TARGET = 'flow[veh/h]'

# =============================================================================
# LOGGING
# =============================================================================

def log_message(message, level=1):
    if not CLI_MODE:
        return
    if VERBOSE_LEVEL >= level:
        print(message)

def log_error(message):
    if CLI_MODE:
        print(f"ERREUR: {message}", file=sys.stderr)
    elif STREAMLIT_AVAILABLE and st is not None:
        st.error(message)

def log_success(message):
    if CLI_MODE:
        log_message(f"✅ {message}")
    elif STREAMLIT_AVAILABLE and st is not None:
        st.success(message)

def log_info(message):
    if CLI_MODE:
        log_message(f"ℹ️  {message}")
    elif STREAMLIT_AVAILABLE and st is not None:
        st.info(message)

def log_warning(message):
    if CLI_MODE:
        log_message(f"⚠️  {message}")
    elif STREAMLIT_AVAILABLE and st is not None:
        st.warning(message)

# =============================================================================
# CHARGEMENT ET NETTOYAGE DES DONNÉES
# =============================================================================

def load_and_clean_cli(csv_files):
    """
    Version CLI de load_and_clean pour traiter une liste de chemins de fichiers.
    
    Args:
        csv_files (list): Liste des chemins vers les fichiers CSV
        
    Returns:
        DataFrame: DataFrame nettoyé et ordonné temporellement
    """
    if not csv_files:
        return pd.DataFrame()
    
    # Lecture et concaténation de tous les fichiers CSV
    df_list = []
    for file_path in csv_files:
        try:
            df_temp = pd.read_csv(file_path, sep=";")
            df_list.append(df_temp)
            log_message(f"Fichier chargé: {os.path.basename(file_path)} ({len(df_temp)} lignes)", 2)
        except Exception as e:
            log_error(f"Erreur lors du chargement de {file_path}: {e}")
            continue
    
    if not df_list:
        log_error("Aucun fichier CSV valide trouvé")
        return pd.DataFrame()
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Nettoyage des IDs de capteurs: conversion en entiers avec gestion des erreurs
    df['count_point_name'] = pd.to_numeric(df['count_point_name'], errors='coerce').astype('Int64')
 
     # Parsing des timestamps
    dt = pd.to_datetime(df['measure_datetime'], errors='coerce', utc=True)
    
    # Tentative de conversion vers le fuseau horaire Europe/Paris
    try:
        dt = dt.dt.tz_convert('Europe/Paris').dt.tz_localize(None)
    except:
        dt = dt.dt.tz_localize(None)
    
    mask = dt.isna() & df['measure_datetime'].notna()
    for i in df[mask].index:
        try:
            dt.at[i] = date_parser.parse(df.at[i, 'measure_datetime'])
        except:
            pass
    
    df['measure_datetime'] = dt
    df.dropna(subset=['measure_datetime', 'count_point_name'], inplace=True)
    df['count_point_name'] = df['count_point_name'].astype(int)
    df.sort_values(['count_point_name', 'measure_datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def load_and_clean_core(csv_files):
    """
    Version core de load_and_clean pour les fichiers uploadés Streamlit.
    """
    df_list = [pd.read_csv(f, sep=";") for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    df['count_point_name'] = pd.to_numeric(df['count_point_name'], errors='coerce').astype('Int64')
    
    dt = pd.to_datetime(df['measure_datetime'], errors='coerce', utc=True)
    
    try:
        dt = dt.dt.tz_convert('Europe/Paris').dt.tz_localize(None)
    except:
        dt = dt.dt.tz_localize(None)
    
    mask = dt.isna() & df['measure_datetime'].notna()
    for i in df[mask].index:
        try:
            dt.at[i] = date_parser.parse(df.at[i, 'measure_datetime'])
        except:
            pass
    
    df['measure_datetime'] = dt
    df.dropna(subset=['measure_datetime', 'count_point_name'], inplace=True)
    df['count_point_name'] = df['count_point_name'].astype(int)
    df.sort_values(['count_point_name', 'measure_datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

if STREAMLIT_AVAILABLE and st is not None:
    @st.cache_data
    def load_and_clean(csv_files):
        return load_and_clean_core(csv_files)
else:
    def load_and_clean(csv_files):
        return load_and_clean_core(csv_files)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def feature_engineering_core(df):
    df2 = df[['count_point_name', 'measure_datetime', 'flow[veh/h]']].copy()
    
    df2['hour_cos'] = np.cos(2 * np.pi * df2['measure_datetime'].dt.hour / 24)
    
    df2['mean_flow_others'] = df2.groupby('measure_datetime')['flow[veh/h]']\
        .transform(lambda x: (x.sum() - x) / (x.count() - 1))
    
    df2['mean_flow_others'] = df2.groupby('count_point_name')['mean_flow_others']\
        .transform(lambda x: x.fillna(x.mean()))
    
    frames = []
    for sid, grp in df2.groupby('count_point_name'):
        g = grp.sort_values('measure_datetime').copy()
        g['ma3'] = g['flow[veh/h]'].rolling(3).mean()
        g['ma6'] = g['flow[veh/h]'].rolling(6).mean()
        g['ma12'] = g['flow[veh/h]'].rolling(12).mean()
        frames.append(g)
    
    df2 = pd.concat(frames, ignore_index=True)
    
    df2.dropna(subset=[
        'flow[veh/h]', 'hour_cos', 'mean_flow_others', 'ma3', 'ma6', 'ma12'
    ], inplace=True)
    
    df2.reset_index(drop=True, inplace=True)
    return df2

def feature_engineering_cli(df):
    return feature_engineering_core(df)

if STREAMLIT_AVAILABLE and st is not None:
    @st.cache_data
    def feature_engineering(df):
        return feature_engineering_core(df)
else:
    def feature_engineering(df):
        return feature_engineering_core(df)

# =============================================================================
# PYTORCH: DATASET ET MODÈLE
# =============================================================================

class TrafficDataset(Dataset):
    def __init__(self, df, feat, target, ws, scaler=None):
        target_in_features = target in feat
        
        if target_in_features:
            cols = feat
            vals = df[cols].values.copy()
            
            if scaler is not None:
                vals = scaler.transform(vals)
            
            X, y = [], []
            for i in range(ws, len(vals)):
                X.append(vals[i-ws:i])
                y.append(df[target].iloc[i])
        else:
            vals = df[feat + [target]].values.copy()
            if scaler is not None:
                vals[:, :-1] = scaler.transform(vals[:, :-1])
            
            X, y = [], []
            for i in range(ws, len(vals)):
                X.append(vals[i-ws:i, :-1])
                y.append(vals[i, -1])
        
        if X:
            self.X = torch.tensor(np.stack(X), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        else:
            n_feat = len(feat)
            self.X = torch.tensor(np.array([]).reshape(0, ws, n_feat), dtype=torch.float32)
            self.y = torch.tensor([]).unsqueeze(-1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hs, nl, do):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hs, nl, batch_first=True, dropout=do if nl > 1 else 0)
        self.fc = nn.Linear(hs, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =============================================================================
# PARSING DES NOMS DE MODÈLES TUNING
# =============================================================================

def parser_nom_modele_tuning(nom_fichier):
    pattern = r'sensor_(.+)_bs(\d+)_hs(\d+)_nl(\d+)_do(\d+)_lr(\d+e[+-]?\d+)_ep(\d+)_ws(\d+)_mae(\d+)\.pt'
    match = re.match(pattern, nom_fichier)
    
    if match:
        return {
            'capteur_inter': match.group(1),
            'batch_size': int(match.group(2)),
            'hidden_size': int(match.group(3)),
            'num_layers': int(match.group(4)),
            'dropout': int(match.group(5)) / 100.0,
            'learning_rate': float(match.group(6)),
            'num_epochs': int(match.group(7)),
            'window_size': int(match.group(8)),
            'mae_original': int(match.group(9))
        }
    return None

# =============================================================================
# CHARGEMENT DE MODÈLES (CLI)
# =============================================================================

def charger_modele_compatible(chemin_modele, feature_cols, device):
    nom_fichier = os.path.basename(chemin_modele)
    
    try:
        ckpt = torch.load(chemin_modele, map_location=device, weights_only=False)
        
        # Checkpoint complet (format lstm.py)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            scaler = ckpt.get('scaler', None)
            features = ckpt.get('feature_set', feature_cols)
            n_features = len(features)
            
            model = LSTMModel(
                input_size=n_features,
                hs=ckpt.get('hidden_size', 64),
                nl=ckpt.get('num_layers', 2),
                do=ckpt.get('dropout', 0.2)
            ).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            
            params = {
                'hidden_size': ckpt.get('hidden_size', 64),
                'num_layers': ckpt.get('num_layers', 2),
                'dropout': ckpt.get('dropout', 0.2),
                'window_size': ckpt.get('window_size', 12),
                'batch_size': ckpt.get('batch_size', 64),
                'lr': ckpt.get('lr', 0.0005),
                'epochs': ckpt.get('epochs', 20),
                'sensor_id': ckpt.get('sensor_id', 'Unknown'),
                'n_features': n_features,
                'feature_set': features
            }
            
            return model, scaler, params
        
        # Modèle tuning (state_dict brut)
        elif isinstance(ckpt, dict) and any(k.startswith('lstm.') or k.startswith('fc.') for k in ckpt.keys()):
            params_from_name = parser_nom_modele_tuning(nom_fichier)
            
            if params_from_name:
                n_features = len(FEATURE_COLS_TUNING)
                
                model = LSTMModel(
                    input_size=n_features,
                    hs=params_from_name['hidden_size'],
                    nl=params_from_name['num_layers'],
                    do=params_from_name['dropout']
                ).to(device)
                model.load_state_dict(ckpt)
                
                params = {
                    'hidden_size': params_from_name['hidden_size'],
                    'num_layers': params_from_name['num_layers'],
                    'dropout': params_from_name['dropout'],
                    'window_size': params_from_name['window_size'],
                    'batch_size': params_from_name['batch_size'],
                    'lr': params_from_name['learning_rate'],
                    'epochs': params_from_name['num_epochs'],
                    'sensor_id': params_from_name['capteur_inter'],
                    'mae_original': params_from_name['mae_original'],
                    'n_features': n_features,
                    'feature_set': FEATURE_COLS_TUNING
                }
                
                return model, None, params
            else:
                raise ValueError(f"Impossible de parser les paramètres du fichier: {nom_fichier}")
        else:
            raise ValueError(f"Format de checkpoint non reconnu pour: {nom_fichier}")
            
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle {chemin_modele}: {e}")

# =============================================================================
# FONCTIONS UTILITAIRES CLI
# =============================================================================

def decouvrir_fichiers_csv(data_folder):
    if not os.path.exists(data_folder):
        return []
    
    csv_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return sorted(csv_files)

def decouvrir_modeles(output_dir):
    model_files = []
    
    model_files.extend(glob.glob("*.pt"))
    
    if output_dir and os.path.exists(output_dir):
        model_files.extend(glob.glob(os.path.join(output_dir, "*.pt")))
    
    if os.path.exists(MODELS_DIR):
        model_files.extend(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    
    seen = set()
    unique_files = []
    for f in model_files:
        abs_path = os.path.abspath(f)
        if abs_path not in seen:
            seen.add(abs_path)
            unique_files.append(f)
    
    return unique_files

def selectionner_fichiers_csv_interactif(csv_files):
    if not csv_files:
        log_error("Aucun fichier CSV trouvé dans le dossier de données.")
        return []
    
    print("\n" + "="*80)
    print("SÉLECTION DES FICHIERS CSV")
    print("="*80)
    
    print(f"\nFichiers CSV disponibles ({len(csv_files)}):")
    for i, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        folder = os.path.dirname(file_path)
        print(f"{i+1:2d}. {filename:30} ({folder})")
    
    print("\nOptions de sélection:")
    print("  - Numéros séparés par des virgules (ex: 1,3,5)")
    print("  - Plages avec tirets (ex: 1-5)")
    print("  - 'all' pour tous les fichiers")
    print("  - 'quit' pour annuler")
    
    while True:
        try:
            choix = input(f"\nVotre sélection (au moins 1 fichier): ").strip()
            
            if choix.lower() == 'quit':
                print("Analyse annulée.")
                sys.exit(0)
            
            if choix.lower() == 'all':
                return csv_files
            
            indices_selectionnes = set()
            for partie in choix.split(','):
                partie = partie.strip()
                if '-' in partie:
                    debut, fin = map(int, partie.split('-'))
                    indices_selectionnes.update(range(debut, fin + 1))
                else:
                    indices_selectionnes.add(int(partie))
            
            indices_valides = [i for i in indices_selectionnes if 1 <= i <= len(csv_files)]
            
            if len(indices_valides) < 1:
                print("⚠️  Veuillez sélectionner au moins 1 fichier.")
                continue
            
            fichiers_selectionnes = [csv_files[i-1] for i in sorted(indices_valides)]
            
            print(f"\n✅ Fichiers sélectionnés ({len(fichiers_selectionnes)}):")
            for file_path in fichiers_selectionnes:
                print(f"   - {os.path.basename(file_path)}")
            
            confirmer = input("\nConfirmer cette sélection? (o/n): ").strip().lower()
            if confirmer in ['o', 'oui', 'y', 'yes']:
                return fichiers_selectionnes
            
        except (ValueError, IndexError) as e:
            print(f"❌ Sélection invalide: {e}")
            print("   Utilisez le format: 1,3,5 ou 1-5 ou 'all'")

def selectionner_capteurs_interactif(all_sids):
    print("\n" + "="*80)
    print("SÉLECTION DES CAPTEURS")
    print("="*80)
    
    print(f"\nCapteurs disponibles ({len(all_sids)}):")
    for i, sid in enumerate(all_sids):
        print(f"{i+1:2d}. Capteur {sid}")
    
    print("\nOptions de sélection:")
    print("  - Numéros séparés par des virgules (ex: 1,3,5)")
    print("  - Plages avec tirets (ex: 1-5)")
    print("  - 'all' pour tous les capteurs")
    print("  - 'quit' pour annuler")
    
    while True:
        try:
            choix = input(f"\nVotre sélection (au moins 1 capteur): ").strip()
            
            if choix.lower() == 'quit':
                print("Analyse annulée.")
                sys.exit(0)
            
            if choix.lower() == 'all':
                return all_sids
            
            indices_selectionnes = set()
            for partie in choix.split(','):
                partie = partie.strip()
                if '-' in partie:
                    debut, fin = map(int, partie.split('-'))
                    indices_selectionnes.update(range(debut, fin + 1))
                else:
                    indices_selectionnes.add(int(partie))
            
            indices_valides = [i for i in indices_selectionnes if 1 <= i <= len(all_sids)]
            
            if len(indices_valides) < 1:
                print("⚠️  Veuillez sélectionner au moins 1 capteur.")
                continue
            
            capteurs_selectionnes = [all_sids[i-1] for i in sorted(indices_valides)]
            
            print(f"\n✅ Capteurs sélectionnés ({len(capteurs_selectionnes)}):")
            for sid in capteurs_selectionnes:
                print(f"   - Capteur {sid}")
            
            confirmer = input("\nConfirmer cette sélection? (o/n): ").strip().lower()
            if confirmer in ['o', 'oui', 'y', 'yes']:
                return capteurs_selectionnes
            
        except (ValueError, IndexError) as e:
            print(f"❌ Sélection invalide: {e}")
            print("   Utilisez le format: 1,3,5 ou 1-5 ou 'all'")

def selectionner_mode_interactif():
    modes = [
        "Entraîner nouveau modèle",
        "Charger modèle existant", 
        "Comparer plusieurs modèles"
    ]
    
    print("\n" + "="*80)
    print("SÉLECTION DU MODE D'UTILISATION")
    print("="*80)
    
    print("\nModes disponibles:")
    for i, mode in enumerate(modes):
        print(f"{i+1}. {mode}")
    
    while True:
        try:
            choix = input(f"\nSélectionnez un mode (1-{len(modes)}): ").strip()
            
            if choix == 'quit':
                print("Analyse annulée.")
                sys.exit(0)
            
            index = int(choix) - 1
            if 0 <= index < len(modes):
                mode_selectionne = modes[index]
                print(f"\n✅ Mode sélectionné: {mode_selectionne}")
                return mode_selectionne
            else:
                print(f"⚠️  Veuillez entrer un numéro entre 1 et {len(modes)}.")
                
        except ValueError:
            print("❌ Veuillez entrer un numéro valide.")

def configurer_hyperparametres_interactif():
    print("\n" + "="*80)
    print("CONFIGURATION DES HYPERPARAMÈTRES LSTM")
    print("="*80)
    
    defaults = {
        'batch_size': 64,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'lr': 0.0005,
        'epochs': 20,
        'window_size': 12
    }
    
    params = {}
    
    print("\nAppuyez sur Entrée pour utiliser les valeurs par défaut entre crochets.")
    
    for param_name, default_value in defaults.items():
        while True:
            try:
                if param_name == 'lr':
                    prompt = f"{param_name} (learning rate) [{default_value}]: "
                    response = input(prompt).strip()
                    if not response:
                        params[param_name] = default_value
                    else:
                        params[param_name] = float(response)
                elif param_name == 'dropout':
                    prompt = f"{param_name} (0.0-0.5) [{default_value}]: "
                    response = input(prompt).strip()
                    if not response:
                        params[param_name] = default_value
                    else:
                        value = float(response)
                        if 0.0 <= value <= 0.5:
                            params[param_name] = value
                        else:
                            print("⚠️  Le dropout doit être entre 0.0 et 0.5.")
                            continue
                else:
                    prompt = f"{param_name} [{default_value}]: "
                    response = input(prompt).strip()
                    if not response:
                        params[param_name] = default_value
                    else:
                        params[param_name] = int(response)
                break
                
            except ValueError:
                print(f"❌ Valeur invalide pour {param_name}. Veuillez réessayer.")
    
    print(f"\n✅ Hyperparamètres configurés:")
    for param_name, value in params.items():
        print(f"   - {param_name}: {value}")
    
    confirmer = input("\nConfirmer cette configuration? (o/n): ").strip().lower()
    if not confirmer in ['o', 'oui', 'y', 'yes']:
        return configurer_hyperparametres_interactif()
    
    return params

def sauvegarder_visualisation_cli(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_success(f"Graphique sauvegardé: {filepath}")

def afficher_tableau_ascii(data, headers, title=None):
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    
    if TABULATE_AVAILABLE:
        print(tabulate(data, headers=headers, tablefmt='grid'))
    else:
        print(f"\n{' | '.join(headers)}")
        print("-" * (len(' | '.join(headers))))
        for row in data:
            print(' | '.join(str(cell) for cell in row))

# =============================================================================
# FONCTIONS PRINCIPALES CLI
# =============================================================================

def run_cli_training(args):
    global CLI_MODE, VERBOSE_LEVEL
    CLI_MODE = True
    VERBOSE_LEVEL = 2 if args.verbose else (0 if args.quiet else 1)
    
    log_message("🚦 PRÉDICTION DE FLUX VÉHICULES - MODE CLI", 1)
    log_message("=" * 50, 1)
    
    data_folder = args.data
    output_dir = args.output
    
    if not os.path.exists(data_folder):
        log_error(f"Dossier de données '{data_folder}' introuvable.")
        sys.exit(1)
    
    log_info(f"Device utilisé: {DEVICE}")
    
    log_message("🔍 Découverte des fichiers CSV...", 1)
    csv_files = decouvrir_fichiers_csv(data_folder)
    
    if not csv_files:
        log_error("Aucun fichier CSV trouvé dans le dossier spécifié.")
        sys.exit(1)
    
    log_info(f"Fichiers CSV trouvés: {len(csv_files)}")
    
    fichiers_selectionnes = selectionner_fichiers_csv_interactif(csv_files)
    
    if not fichiers_selectionnes:
        log_error("Aucun fichier sélectionné.")
        sys.exit(1)
    
    log_message("📁 Chargement et nettoyage des données...", 1)
    df = load_and_clean_cli(fichiers_selectionnes)
    
    if df.empty:
        log_error("Aucune donnée valide après nettoyage.")
        sys.exit(1)
    
    log_success(f"Données chargées: {len(df)} lignes")
    
    all_sids = sorted(df['count_point_name'].unique())
    log_info(f"Capteurs disponibles: {len(all_sids)}")
    
    sids = selectionner_capteurs_interactif(all_sids)
    
    if not sids:
        log_error("Aucun capteur sélectionné.")
        sys.exit(1)
    
    df = df[df['count_point_name'].isin(sids)].reset_index(drop=True)
    log_success(f"Données filtrées: {len(df)} lignes pour {len(sids)} capteurs")
    
    mode = selectionner_mode_interactif()
    
    log_message("🔧 Génération des caractéristiques...", 1)
    df = feature_engineering_cli(df)
    
    FEATURE_COLS = FEATURE_COLS_TUNING
    log_success(f"Features créées: {len(df)} lignes avec {len(FEATURE_COLS)} caractéristiques")
    
    if mode == "Entraîner nouveau modèle":
        run_cli_train_new_model(df, sids, FEATURE_COLS, TARGET, output_dir)
    elif mode == "Charger modèle existant":
        run_cli_load_existing_model(df, sids, FEATURE_COLS, TARGET, output_dir)
    elif mode == "Comparer plusieurs modèles":
        run_cli_compare_models(df, sids, FEATURE_COLS, TARGET, output_dir)

def run_cli_train_new_model(df, sids, FEATURE_COLS, TARGET, output_dir):
    log_message("🏗️ MODE: ENTRAÎNEMENT DE NOUVEAUX MODÈLES", 1)
    log_message("-" * 45, 1)
    
    params = configurer_hyperparametres_interactif()
    
    batch_size = params['batch_size']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    dropout = params['dropout']
    lr = params['lr']
    epochs = params['epochs']
    window_size = params['window_size']
    
    cv_enabled = True
    
    log_message("▶ Lancement de l'entraînement...", 1)
    all_metrics = []
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for sid in sids:
        log_message(f"🔧 Entraînement du capteur {sid}", 1)
        
        grp = df[df['count_point_name'] == sid].reset_index(drop=True)
        cut = int(0.8 * len(grp))
        train_df, test_df = grp.iloc[:cut], grp.iloc[cut:]
        
        log_message(f"Données d'entraînement: {len(train_df)} lignes", 2)
        log_message(f"Données de test: {len(test_df)} lignes", 2)

        scaler = None
        if FEATURE_COLS:
            scaler = StandardScaler().fit(train_df[FEATURE_COLS])

        ds_tr = TrafficDataset(train_df, FEATURE_COLS, TARGET, window_size, scaler)
        ds_te = TrafficDataset(test_df, FEATURE_COLS, TARGET, window_size, scaler)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        model = LSTMModel(len(FEATURE_COLS), hidden_size, num_layers, dropout).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_losses = []
        model.train()
        
        log_message(f"Entraînement sur {epochs} époques...", 2)
        for epoch in range(1, epochs+1):
            cum = 0.0
            
            for Xb, yb in dl_tr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(Xb), yb)
                loss.backward()
                opt.step()
                cum += loss.item() * Xb.size(0)
            
            epoch_loss = cum / len(ds_tr) if len(ds_tr) > 0 else 0
            train_losses.append(epoch_loss)
            
            if VERBOSE_LEVEL >= 2 and epoch % 5 == 0:
                log_message(f"  Époque {epoch:2d}/{epochs}: Loss = {epoch_loss:.4f}", 2)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(range(1, epochs+1), train_losses, marker='o')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        ax1.set_title(f"Capteur {sid} – Évolution de la Loss pendant l'Entraînement")
        ax1.grid(True, alpha=0.3)
        sauvegarder_visualisation_cli(fig1, f"loss_capteur_{sid}_{timestamp}", output_dir)

        if cv_enabled:
            log_message("Validation croisée en cours...", 2)
            tscv = TimeSeriesSplit(n_splits=5)
            mses, maes = [], []
            
            for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_df), start=1):
                log_message(f"  Fold {fold}/5", 2)
                tr_f = train_df.iloc[tr_idx]
                vl_f = train_df.iloc[val_idx]
                
                mdl = LSTMModel(len(FEATURE_COLS), hidden_size, num_layers, dropout).to(DEVICE)
                opt2 = optim.Adam(mdl.parameters(), lr=lr)
                
                ds_tr_f = TrafficDataset(tr_f, FEATURE_COLS, TARGET, window_size, scaler)
                dl_tr_f = DataLoader(ds_tr_f, batch_size=batch_size, shuffle=True)
                
                for _ in range(min(10, epochs)):
                    for Xb, yb in dl_tr_f:
                        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                        opt2.zero_grad()
                        nn.MSELoss()(mdl(Xb), yb).backward()
                        opt2.step()
                
                ds_vl_f = TrafficDataset(vl_f, FEATURE_COLS, TARGET, window_size, scaler)
                dl_vl_f = DataLoader(ds_vl_f, batch_size=batch_size, shuffle=False)
                
                mse_f = mae_f = n_f = 0
                with torch.no_grad():
                    for Xb, yb in dl_vl_f:
                        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                        p = mdl(Xb)
                        mse_f += ((p - yb) ** 2).sum().item()
                        mae_f += (p - yb).abs().sum().item()
                        n_f += yb.numel()
                
                if n_f > 0:
                    mses.append(mse_f / n_f)
                    maes.append(mae_f / n_f)
            
            cv_mean_mse, cv_std_mse = (np.mean(mses), np.std(mses)) if mses else (0, 0)
            cv_mean_mae, cv_std_mae = (np.mean(maes), np.std(maes)) if maes else (0, 0)
            
            log_message(f"CV MSE: {cv_mean_mse:.2f} ± {cv_std_mse:.2f}", 1)
            log_message(f"CV MAE: {cv_mean_mae:.2f} ± {cv_std_mae:.2f}", 1)
        else:
            cv_mean_mse = cv_std_mse = cv_mean_mae = cv_std_mae = None

        model.eval()
        preds, actuals = [], []
        
        with torch.no_grad():
            for Xb, yb in dl_te:
                Xb = Xb.to(DEVICE)
                out = model(Xb).cpu().squeeze().tolist()
                if isinstance(out, float):
                    out = [out]
                preds.extend(out)
                actuals.extend(yb.squeeze().tolist())
        
        mse_test = np.mean((np.array(preds) - actuals) ** 2) if preds else 0
        mae_test = np.mean(np.abs(np.array(preds) - actuals)) if preds else 0
        
        log_message(f"Test MSE: {mse_test:.2f} — MAE: {mae_test:.2f}", 1)

        if preds:
            dfp = pd.DataFrame({
                'datetime': pd.to_datetime(test_df['measure_datetime'].values[window_size:]),
                'Réel': actuals,
                'Prédit': preds
            }).set_index('datetime')
            
            if not dfp.empty:
                date_exemple = dfp.index.date[len(dfp)//2]
                df_day = dfp[dfp.index.date == date_exemple]
                
                if not df_day.empty:
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    df_day.plot(ax=ax2)
                    ax2.set_ylabel("Flux (veh/h)")
                    ax2.set_title(f"Capteur {sid} – Réel vs Prédit le {date_exemple}")
                    ax2.set_xlabel("Heure")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    sauvegarder_visualisation_cli(fig2, f"predictions_capteur_{sid}_{date_exemple}_{timestamp}", output_dir)

        ckpt = {
            'sensor_id': sid,
            'feature_set': FEATURE_COLS,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'lr': lr,
            'epochs': epochs,
            'window_size': window_size,
            'train_losses': train_losses,
            'cv_mean_mse': cv_mean_mse,
            'cv_std_mse': cv_std_mse,
            'cv_mean_mae': cv_mean_mae,
            'cv_std_mae': cv_std_mae,
            'scaler': scaler,
            'model_state_dict': model.state_dict()
        }
        
        fname = (
            f"lstm_sensor_{sid}"
            f"_bs{batch_size}"
            f"_hs{hidden_size}"
            f"_nl{num_layers}"
            f"_do{int(dropout*100)}"
            f"_lr{lr:.0e}"
            f"_ep{epochs}"
            f"_ws{window_size}"
            ".pt"
        )
        
        model_path = os.path.join(output_dir, fname)
        torch.save(ckpt, model_path)
        log_success(f"Modèle sauvegardé: {model_path}")

        all_metrics.append({
            'sensor': sid,
            'Test MSE': round(mse_test, 2),
            'Test MAE': round(mae_test, 2),
            'CV MSE μ': round(cv_mean_mse, 2) if cv_enabled and cv_mean_mse is not None else "–",
            'CV MAE μ': round(cv_mean_mae, 2) if cv_enabled and cv_mean_mae is not None else "–"
        })

    log_message("📊 RÉCAPITULATIF DES CAPTEURS", 1)
    log_message("=" * 35, 1)
    
    table_data = []
    for metric in all_metrics:
        table_data.append([
            f"Capteur {metric['sensor']}",
            metric['Test MSE'],
            metric['Test MAE'],
            metric['CV MSE μ'],
            metric['CV MAE μ']
        ])
    
    afficher_tableau_ascii(
        table_data,
        ['Capteur', 'Test MSE', 'Test MAE', 'CV MSE μ', 'CV MAE μ'],
        "Performances Finales des Modèles"
    )

    log_success("🎉 Entraînement terminé pour tous les capteurs sélectionnés !")
    log_info(f"💾 Modèles et visualisations sauvegardés dans: {output_dir}")

def run_cli_load_existing_model(df, sids, FEATURE_COLS, TARGET, output_dir):
    log_message("📂 MODE: CHARGEMENT DE MODÈLES EXISTANTS", 1)
    log_message("-" * 42, 1)
    
    model_files = decouvrir_modeles(output_dir)
    
    if not model_files:
        log_error("Aucun modèle .pt trouvé dans:")
        log_error("  - Répertoire courant")
        log_error(f"  - {output_dir}")
        log_error(f"  - {MODELS_DIR}")
        log_info("Conseil: Lancez d'abord un entraînement ou vérifiez l'emplacement des modèles.")
        return
    
    log_info(f"Modèles trouvés: {len(model_files)}")
    
    print("\nModèles disponibles:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1:2d}. {model_file}")
    
    models_to_load = {}
    for sid in sids:
        print(f"\nSélectionnez le modèle pour le capteur {sid}:")
        print("0. Ignorer ce capteur")
        
        while True:
            try:
                choix = input(f"Votre choix (0-{len(model_files)}): ").strip()
                
                if choix == '0':
                    break
                
                index = int(choix) - 1
                if 0 <= index < len(model_files):
                    models_to_load[sid] = model_files[index]
                    log_success(f"Modèle sélectionné pour capteur {sid}: {os.path.basename(model_files[index])}")
                    break
                else:
                    print(f"⚠️  Veuillez entrer un numéro entre 0 et {len(model_files)}.")
                    
            except ValueError:
                print("❌ Veuillez entrer un numéro valide.")
    
    if not models_to_load:
        log_warning("Aucun modèle sélectionné.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    for sid, model_file in models_to_load.items():
        log_message(f"🔍 Évaluation du capteur {sid}", 1)
        
        try:
            model, scaler, params = charger_modele_compatible(model_file, FEATURE_COLS, DEVICE)
            model.eval()
            
            window_size = params.get('window_size', 12)
            model_features = params.get('feature_set', FEATURE_COLS)
            n_features = params.get('n_features', len(model_features))
            
            log_message(f"  Modèle avec {n_features} features: {model_features}", 2)
            
            grp = df[df['count_point_name'] == sid].reset_index(drop=True)
            cut = int(0.8 * len(grp))
            test_df = grp.iloc[cut:].reset_index(drop=True)
            
            adapted_features = []
            for f in model_features:
                if f == 'flow' and 'flow' not in test_df.columns and 'flow[veh/h]' in test_df.columns:
                    adapted_features.append('flow[veh/h]')
                elif f == 'flow[veh/h]' and 'flow[veh/h]' not in test_df.columns and 'flow' in test_df.columns:
                    adapted_features.append('flow')
                else:
                    adapted_features.append(f)
            
            ds_te = TrafficDataset(test_df, adapted_features, TARGET, window_size, scaler)
            dl_te = DataLoader(ds_te, batch_size=64, shuffle=False)
            
            preds, actuals = [], []
            with torch.no_grad():
                for Xb, yb in dl_te:
                    Xb = Xb.to(DEVICE)
                    p = model(Xb).cpu().squeeze().tolist()
                    if isinstance(p, float):
                        p = [p]
                    preds.extend(p)
                    actuals.extend(yb.squeeze().tolist())
            
            mse = np.mean((np.array(preds) - actuals) ** 2) if preds else 0
            mae = np.mean(np.abs(np.array(preds) - actuals)) if preds else 0
            
            mean_flow = test_df[TARGET].mean()
            mae_pct = 100.0 * mae / mean_flow if mean_flow > 0 else 0
            
            result_info = {
                'Capteur': sid,
                'Modèle': os.path.basename(model_file),
                'MSE': round(mse, 2),
                'MAE': round(mae, 2),
                'MAE%': round(mae_pct, 2),
                'Hidden_Size': params.get('hidden_size', 'N/A'),
                'Num_Layers': params.get('num_layers', 'N/A'),
                'Dropout': params.get('dropout', 'N/A'),
                'Window_Size': window_size,
                'N_Features': n_features
            }
            results.append(result_info)
            
            log_message(f"Test MSE: {mse:.2f} — MAE: {mae:.2f} ({mae_pct:.2f}%)", 1)
            
            if preds:
                dfp = pd.DataFrame({
                    'datetime': pd.to_datetime(test_df['measure_datetime'].values[window_size:]),
                    'Réel': actuals,
                    'Prédit': preds
                }).set_index('datetime')
                
                if not dfp.empty:
                    date_exemple = dfp.index.date[len(dfp)//2]
                    df_day = dfp[dfp.index.date == date_exemple]
                    
                    if not df_day.empty:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        df_day.plot(ax=ax)
                        ax.set_title(f"Capteur {sid} – Réel vs Prédit le {date_exemple}")
                        ax.set_xlabel("Heure")
                        ax.set_ylabel("Flux (veh/h)")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        sauvegarder_visualisation_cli(fig, f"evaluation_capteur_{sid}_{date_exemple}_{timestamp}", output_dir)
            
        except Exception as e:
            log_error(f"Erreur lors de l'évaluation du modèle pour le capteur {sid}: {e}")
            import traceback
            log_message(traceback.format_exc(), 2)
            continue
    
    if results:
        log_message("📊 RÉSULTATS DE L'ÉVALUATION", 1)
        log_message("=" * 30, 1)
        
        table_data = []
        for result in results:
            table_data.append([
                result['Capteur'],
                result['MSE'],
                result['MAE'],
                f"{result['MAE%']}%",
                result['Hidden_Size'],
                result['Num_Layers'],
                f"{result['Dropout']:.2f}" if isinstance(result['Dropout'], float) else result['Dropout'],
                result['Window_Size']
            ])
        
        afficher_tableau_ascii(
            table_data,
            ['Capteur', 'MSE', 'MAE', 'MAE%', 'Hidden', 'Layers', 'Dropout', 'Window'],
            "Performances des Modèles Chargés"
        )
        
        results_df = pd.DataFrame(results)
        results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        log_success(f"Résultats sauvegardés: {results_file}")
    
    log_success("🎉 Évaluation terminée !")

def run_cli_compare_models(df, sids, FEATURE_COLS, TARGET, output_dir):
    log_message("⚖️ MODE: COMPARAISON DE PLUSIEURS MODÈLES", 1)
    log_message("-" * 42, 1)
    
    model_files = decouvrir_modeles(output_dir)
    
    if len(model_files) < 2:
        log_error("Au moins 2 modèles .pt sont nécessaires pour la comparaison.")
        log_info("Modèles recherchés dans:")
        log_info("  - Répertoire courant")
        log_info(f"  - {output_dir}")
        log_info(f"  - {MODELS_DIR}")
        return
    
    log_info(f"Modèles disponibles: {len(model_files)}")
    
    print("\nModèles disponibles:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1:2d}. {model_file}")
    
    print("\nOptions de sélection:")
    print("  - Numéros séparés par des virgules (ex: 1,3,5)")
    print("  - Plages avec tirets (ex: 1-5)")
    print("  - 'all' pour tous les modèles")
    
    while True:
        try:
            choix = input(f"\nSélectionnez les modèles à comparer (au moins 2): ").strip()
            
            if choix.lower() == 'all':
                models_to_compare = model_files
                break
            
            indices_selectionnes = set()
            
            for partie in choix.split(','):
                partie = partie.strip()
                if '-' in partie:
                    debut, fin = map(int, partie.split('-'))
                    indices_selectionnes.update(range(debut, fin + 1))
                else:
                    indices_selectionnes.add(int(partie))
            
            indices_valides = [i for i in indices_selectionnes if 1 <= i <= len(model_files)]
            
            if len(indices_valides) < 2:
                print("⚠️  Veuillez sélectionner au moins 2 modèles.")
                continue
            
            models_to_compare = [model_files[i-1] for i in sorted(indices_valides)]
            break
            
        except (ValueError, IndexError) as e:
            print(f"❌ Sélection invalide: {e}")
    
    log_success(f"Modèles sélectionnés pour comparaison: {len(models_to_compare)}")
    for model_file in models_to_compare:
        log_message(f"  - {os.path.basename(model_file)}", 1)
    
    sensor_ids = set()
    model_infos = []
    
    for model_file in models_to_compare:
        try:
            model, scaler, params = charger_modele_compatible(model_file, FEATURE_COLS, DEVICE)
            sensor_id = params.get('sensor_id', 'Unknown')
            sensor_ids.add(sensor_id)
            model_infos.append({
                'file': model_file,
                'sensor_id': sensor_id,
                'model': model,
                'scaler': scaler,
                'params': params
            })
        except Exception as e:
            log_error(f"Erreur lors du chargement de {model_file}: {e}")
            continue
    
    if len(sensor_ids) != 1:
        log_warning(f"Les modèles concernent différents capteurs: {sensor_ids}")
        log_info("La comparaison sera effectuée sur le premier capteur trouvé dans les données.")
    
    target_sensor = None
    for sid in sensor_ids:
        if sid in sids or (isinstance(sid, str) and any(str(s) in str(sid) for s in sids)):
            for s in sids:
                if str(s) in str(sid):
                    target_sensor = s
                    break
            if target_sensor:
                break
    
    if target_sensor is None:
        target_sensor = sids[0] if sids else list(sensor_ids)[0]
    
    log_info(f"Comparaison des modèles sur le capteur: {target_sensor}")
    
    grp = df[df['count_point_name'] == target_sensor].reset_index(drop=True)
    if grp.empty:
        log_error(f"Aucune donnée trouvée pour le capteur {target_sensor}")
        return
    
    cut = int(0.8 * len(grp))
    test_df = grp.iloc[cut:].reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    comparison_results = []
    predictions_data = {}
    
    for i, model_info in enumerate(model_infos, 1):
        model_file = model_info['file']
        model = model_info['model']
        scaler = model_info['scaler']
        params = model_info['params']
        
        log_message(f"🔍 Évaluation du modèle {i}: {os.path.basename(model_file)}", 1)
        
        try:
            model.eval()
            window_size = params.get('window_size', 12)
            model_features = params.get('feature_set', FEATURE_COLS)
            
            adapted_features = []
            for f in model_features:
                if f == 'flow' and 'flow' not in test_df.columns and 'flow[veh/h]' in test_df.columns:
                    adapted_features.append('flow[veh/h]')
                elif f == 'flow[veh/h]' and 'flow[veh/h]' not in test_df.columns and 'flow' in test_df.columns:
                    adapted_features.append('flow')
                else:
                    adapted_features.append(f)
            
            ds_te = TrafficDataset(test_df, adapted_features, TARGET, window_size, scaler)
            dl_te = DataLoader(ds_te, batch_size=64, shuffle=False)
            
            preds, actuals = [], []
            with torch.no_grad():
                for Xb, yb in dl_te:
                    p = model(Xb.to(DEVICE)).cpu().squeeze().tolist()
                    if isinstance(p, float):
                        p = [p]
                    preds.extend(p)
                    actuals.extend(yb.squeeze().tolist())
            
            mse = np.mean((np.array(preds) - actuals) ** 2) if preds else 0
            mae = np.mean(np.abs(np.array(preds) - actuals)) if preds else 0
            
            comparison_results.append({
                'Modèle': f"Modèle_{i}",
                'Fichier': os.path.basename(model_file),
                'MSE': round(mse, 2),
                'MAE': round(mae, 2),
                'Hidden_Size': params.get('hidden_size', 'N/A'),
                'Num_Layers': params.get('num_layers', 'N/A'),
                'Dropout': params.get('dropout', 'N/A'),
                'Learning_Rate': params.get('lr', 'N/A'),
                'Epochs': params.get('epochs', 'N/A'),
                'Window_Size': window_size
            })
            
            predictions_data[f"Modèle_{i}"] = {
                'preds': preds,
                'actuals': actuals,
                'window_size': window_size
            }
            
            log_message(f"MSE: {mse:.2f}, MAE: {mae:.2f}", 2)
            
        except Exception as e:
            log_error(f"Erreur lors de l'évaluation de {model_file}: {e}")
            continue
    
    if not comparison_results:
        log_error("Aucun modèle n'a pu être évalué avec succès.")
        return
    
    log_message("📊 RÉSULTATS DE LA COMPARAISON", 1)
    log_message("=" * 35, 1)
    
    table_data = []
    for result in comparison_results:
        table_data.append([
            result['Modèle'],
            result['MSE'],
            result['MAE'],
            result['Hidden_Size'],
            result['Num_Layers'],
            f"{result['Dropout']:.2f}" if isinstance(result['Dropout'], float) else result['Dropout'],
            result['Window_Size']
        ])
    
    afficher_tableau_ascii(
        table_data,
        ['Modèle', 'MSE', 'MAE', 'Hidden', 'Layers', 'Dropout', 'Window'],
        "Comparaison des Performances"
    )
    
    best_model = min(comparison_results, key=lambda x: x['MAE'])
    log_success(f"🏆 Meilleur modèle (MAE la plus faible): {best_model['Modèle']} (MAE: {best_model['MAE']})")
    
    if len(predictions_data) >= 2:
        first_model = list(predictions_data.keys())[0]
        actuals = predictions_data[first_model]['actuals']
        window_size = predictions_data[first_model]['window_size']
        
        if actuals:
            plot_data = {
                'datetime': pd.to_datetime(test_df['measure_datetime'].values[window_size:]),
                'Réel': actuals
            }
            
            for model_name, data in predictions_data.items():
                plot_data[model_name] = data['preds']
            
            dfp = pd.DataFrame(plot_data).set_index('datetime')
            
            if not dfp.empty:
                date_exemple = dfp.index.date[len(dfp)//2]
                df_day = dfp[dfp.index.date == date_exemple]
                
                if not df_day.empty:
                    fig, ax = plt.subplots(figsize=(14, 8))
                    df_day.plot(ax=ax, linewidth=2)
                    ax.set_title(f"Comparaison des Modèles - Capteur {target_sensor} le {date_exemple}", 
                                 fontsize=16, fontweight='bold')
                    ax.set_xlabel("Heure", fontsize=12)
                    ax.set_ylabel("Flux (veh/h)", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=10)
                    sauvegarder_visualisation_cli(fig, f"comparaison_modeles_capteur_{target_sensor}_{date_exemple}_{timestamp}", output_dir)
    
    results_df = pd.DataFrame(comparison_results)
    results_file = os.path.join(output_dir, f"comparison_results_{target_sensor}_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    log_success(f"Résultats détaillés sauvegardés: {results_file}")
    
    log_success("🎉 Comparaison terminée !")

# =============================================================================
# CHARGEMENT DE MODÈLES (STREAMLIT)
# =============================================================================

def charger_modele_compatible_streamlit(f, feature_cols_default, device):
    import io

    bytes_data = f.read()
    f.seek(0)
    buffer = io.BytesIO(bytes_data)

    ckpt = torch.load(buffer, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        scaler = ckpt.get('scaler', None)
        features = ckpt.get('feature_set', feature_cols_default)
        n_feat = len(features)

        model = LSTMModel(
            input_size=n_feat,
            hs=ckpt.get('hidden_size', 64),
            nl=ckpt.get('num_layers', 2),
            do=ckpt.get('dropout', 0.2)
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        params = {
            'window_size': ckpt.get('window_size', 12),
            'feature_set': features,
            'hidden_size': ckpt.get('hidden_size', 64),
            'num_layers': ckpt.get('num_layers', 2),
            'dropout': ckpt.get('dropout', 0.2),
            'sensor_id': ckpt.get('sensor_id', 'Unknown')
        }
        return model, scaler, params

    else:
        params_name = parser_nom_modele_tuning(f.name)
        if params_name:
            features_tuning = FEATURE_COLS_TUNING

            model = LSTMModel(
                input_size=len(features_tuning),
                hs=params_name['hidden_size'],
                nl=params_name['num_layers'],
                do=params_name['dropout']
            ).to(device)
            model.load_state_dict(ckpt)

            params = {
                'window_size': params_name['window_size'],
                'feature_set': features_tuning,
                'hidden_size': params_name['hidden_size'],
                'num_layers': params_name['num_layers'],
                'dropout': params_name['dropout'],
                'mae_original': params_name['mae_original'],
                'sensor_id': params_name['capteur_inter']
            }
            return model, None, params

        raise ValueError("Impossible de reconnaître le format du modèle (ni standard, ni tuning)")

# =============================================================================
# MAIN STREAMLIT (GUI)
# =============================================================================

def main_streamlit():
    if not STREAMLIT_AVAILABLE or st is None:
        print("Erreur: Streamlit n'est pas disponible.")
        sys.exit(1)
    
    st.set_page_config(page_title="Traffic Flow Predictor", layout="wide")
    st.title("🚦 Prédiction de flux véhicules")
    
    st.sidebar.header("1. Chargement des données")
    uploaded = st.sidebar.file_uploader(
        "Sélectionnez un ou plusieurs fichiers CSV",
        type="csv", accept_multiple_files=True
    )

    if not uploaded:
        st.warning("☝️ Veuillez charger au moins un CSV pour continuer")
        st.stop()

    df = load_and_clean(uploaded)

    st.sidebar.header("2. Choix des capteurs")
    all_sids = sorted(df['count_point_name'].unique())
    sids = st.sidebar.multiselect("Sélectionnez au moins un capteur", all_sids)

    if not sids:
        st.warning("☝️ Choisissez au moins un capteur")
        st.stop()

    df = df[df['count_point_name'].isin(sids)].reset_index(drop=True)

    mode = st.sidebar.radio(
        "3. Mode d'utilisation",
        ["Entraîner nouveau modèle", "Charger modèle existant", "Comparer plusieurs modèles"]
    )

    df = feature_engineering(df)

    FEATURE_COLS = FEATURE_COLS_LSTM

    # MODE: CHARGER MODÈLE EXISTANT
    if mode == "Charger modèle existant":
        st.sidebar.header("4. Charger des modèles `.pt`")

        uploaders = {
            sid: st.sidebar.file_uploader(
                f"Modèle pour capteur {sid}", type="pt", key=f"mdl_{sid}"
            )
            for sid in sids
        }

        if not any(uploaders.values()):
            st.info("Charge au moins un modèle `.pt` dans la barre latérale.")
            st.stop()

        st.header("🔍 Évaluation des modèles chargés")

        for sid, f_mdl in uploaders.items():
            if f_mdl is None:
                continue

            st.subheader(f"Capteur {sid}")

            try:
                model, scaler, params = charger_modele_compatible_streamlit(
                    f_mdl, FEATURE_COLS, DEVICE
                )
                model.eval()

                model_features = params['feature_set']
                ws = params['window_size']

                grp = df[df['count_point_name'] == sid].reset_index(drop=True)
                cut = int(0.8 * len(grp))
                test_df = grp.iloc[cut:].reset_index(drop=True)

                ds_te = TrafficDataset(test_df, model_features, TARGET, ws, scaler)
                dl_te = DataLoader(ds_te, batch_size=64, shuffle=False)

                preds, acts = [], []
                with torch.no_grad():
                    for Xb, yb in dl_te:
                        p = model(Xb.to(DEVICE)).cpu().squeeze().tolist()
                        if isinstance(p, float):
                            p = [p]
                        preds.extend(p)
                        acts.extend(yb.squeeze().tolist())

                if not preds:
                    st.warning("Pas assez de données pour évaluer ce modèle.")
                    continue

                preds_arr = np.array(preds)
                acts_arr = np.array(acts)

                mse = np.mean((preds_arr - acts_arr) ** 2)
                mae = np.mean(np.abs(preds_arr - acts_arr))

                st.write(f"**Test MSE**: {mse:.2f} — **MAE**: {mae:.2f}")
                st.expander("Paramètres du modèle").write(params)

                df_res = pd.DataFrame({
                    'datetime': pd.to_datetime(test_df['measure_datetime'].values[ws:]),
                    'Réel': acts_arr,
                    'Prédit': preds_arr
                }).set_index('datetime')

                if df_res.empty:
                    st.warning("Pas de points à tracer.")
                    continue

                st.line_chart(df_res.tail(200))

            except Exception as e:
                st.error(f"Erreur pour le capteur {sid}: {e}")

        st.stop()

    # MODE: COMPARER PLUSIEURS MODÈLES
    if mode == "Comparer plusieurs modèles":
        st.sidebar.header("4. Charger plusieurs modèles `.pt`")

        uploaded_models = st.sidebar.file_uploader(
            "Sélectionne au moins 2 modèles", type="pt", accept_multiple_files=True
        )

        if not uploaded_models or len(uploaded_models) < 2:
            st.warning("☝️ Charge au moins 2 fichiers `.pt` pour comparer")
            st.stop()

        st.header(f"⚖️ Comparaison de {len(uploaded_models)} modèles")
        st.info("La comparaison est effectuée sur le premier capteur sélectionné.")

        sid = sids[0]
        grp = df[df['count_point_name'] == sid].reset_index(drop=True)
        cut = int(0.8 * len(grp))
        test_df = grp.iloc[cut:].reset_index(drop=True)

        results = []
        curves = {}

        for i, f_mdl in enumerate(uploaded_models, start=1):
            try:
                model, scaler, params = charger_modele_compatible_streamlit(
                    f_mdl, FEATURE_COLS, DEVICE
                )
                model.eval()

                model_features = params['feature_set']
                ws = params['window_size']

                ds_te = TrafficDataset(test_df, model_features, TARGET, ws, scaler)
                dl_te = DataLoader(ds_te, batch_size=64, shuffle=False)

                preds, acts = [], []
                with torch.no_grad():
                    for Xb, yb in dl_te:
                        p = model(Xb.to(DEVICE)).cpu().squeeze().tolist()
                        if isinstance(p, float):
                            p = [p]
                        preds.extend(p)
                        acts.extend(yb.squeeze().tolist())

                if not preds:
                    st.warning(f"Pas assez de données pour le modèle {f_mdl.name}")
                    continue

                preds_arr = np.array(preds)
                acts_arr = np.array(acts)

                mse = np.mean((preds_arr - acts_arr) ** 2)
                mae = np.mean(np.abs(preds_arr - acts_arr))

                results.append({
                    'Modèle': f"Modèle {i}",
                    'Fichier': f_mdl.name,
                    'MSE': round(mse, 2),
                    'MAE': round(mae, 2),
                    **params
                })

                curves[f"Modèle {i}"] = (preds_arr, acts_arr, ws)

            except Exception as e:
                st.error(f"Erreur avec {f_mdl.name}: {e}")

        if not results:
            st.warning("Aucun modèle n'a pu être évalué.")
            st.stop()

        st.dataframe(pd.DataFrame(results))
        best = min(results, key=lambda x: x['MAE'])
        st.success(
            f"🏆 Meilleur modèle: {best['Modèle']} ({best['Fichier']}) "
            f"avec MAE = {best['MAE']}"
        )

        first_key = next(iter(curves))
        preds0, acts0, ws0 = curves[first_key]

        df_base = pd.DataFrame({
            'datetime': pd.to_datetime(test_df['measure_datetime'].values[ws0:]),
            'Réel': acts0
        }).set_index('datetime')

        for name, (preds_arr, _, ws) in curves.items():
            if ws == ws0:
                df_base[name] = preds_arr

        st.line_chart(df_base.tail(200))

        st.stop()

    # MODE: ENTRAÎNEMENT
    st.sidebar.header("4. Hyperparamètres LSTM")

    batch_size = st.sidebar.number_input("Batch size", 8, 512, 64, step=8)
    hidden_size = st.sidebar.number_input("Hidden size", 8, 512, 64, step=8)
    num_layers = st.sidebar.number_input("Nb de couches", 1, 4, 2, step=1)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
    lr = st.sidebar.slider(
        "Learning rate", 1e-4, 1e-2, value=5e-4, step=1e-5, format="%.5f"
    )
    epochs = st.sidebar.number_input("Époques", 1, 100, 20, step=1)
    window_size = st.sidebar.number_input("Window size", 1, 48, 12, step=1)
    cv_enabled = st.sidebar.checkbox("Validation croisée (5 folds)", value=False)

    if st.sidebar.button("▶ Lancer l'entraînement"):
        st.sidebar.success("Entraînement en cours…")
        all_metrics = []

        for sid in sids:
            st.subheader(f"🔧 Capteur {sid}")
            
            grp = df[df['count_point_name'] == sid].reset_index(drop=True)
            cut = int(0.8 * len(grp))
            train_df, test_df = grp.iloc[:cut], grp.iloc[cut:]
            
            scaler = StandardScaler().fit(train_df[FEATURE_COLS])
            
            ds_tr = TrafficDataset(train_df, FEATURE_COLS, TARGET, window_size, scaler)
            ds_te = TrafficDataset(test_df, FEATURE_COLS, TARGET, window_size, scaler)
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
            dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
            
            model = LSTMModel(len(FEATURE_COLS), hidden_size, num_layers, dropout).to(DEVICE)
            opt = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            
            train_losses = []
            progress = st.progress(0)
            
            for epoch in range(1, epochs+1):
                model.train()
                cum = 0.0
                for Xb, yb in dl_tr:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss = loss_fn(model(Xb), yb)
                    loss.backward()
                    opt.step()
                    cum += loss.item() * Xb.size(0)
                
                epoch_loss = cum / len(ds_tr) if len(ds_tr) > 0 else 0
                train_losses.append(epoch_loss)
                progress.progress(epoch / epochs)
            
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for Xb, yb in dl_te:
                    out = model(Xb.to(DEVICE)).cpu().squeeze().tolist()
                    if isinstance(out, float):
                        out = [out]
                    preds.extend(out)
                    actuals.extend(yb.squeeze().tolist())
            
            if preds:
                mse = np.mean((np.array(preds) - actuals) ** 2)
                mae = np.mean(np.abs(np.array(preds) - actuals))
                st.write(f"**Test MSE**: {mse:.2f} — **MAE**: {mae:.2f}")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(train_losses)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Évolution de la Loss - Capteur {sid}")
                st.pyplot(fig)
                
                ckpt = {
                    'sensor_id': sid,
                    'feature_set': FEATURE_COLS,
                    'batch_size': batch_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'lr': lr,
                    'epochs': epochs,
                    'window_size': window_size,
                    'train_losses': train_losses,
                    'scaler': scaler,
                    'model_state_dict': model.state_dict()
                }
                
                import io
                buffer = io.BytesIO()
                torch.save(ckpt, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label=f"💾 Télécharger modèle capteur {sid}",
                    data=buffer,
                    file_name=f"lstm_sensor_{sid}_hs{hidden_size}_nl{num_layers}.pt",
                    mime="application/octet-stream"
                )
                
                all_metrics.append({
                    'Capteur': sid,
                    'MSE': round(mse, 2),
                    'MAE': round(mae, 2)
                })
        
        if all_metrics:
            st.header("📊 Récapitulatif")
            st.dataframe(pd.DataFrame(all_metrics))

# =============================================================================
# MAIN (ENTRY POINT)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Application de prédiction de flux véhicules avec modèles LSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  Mode GUI (Streamlit):
    python lstm.py

  Mode CLI:
    python lstm.py --cli --data ./data --output ./resultats
    
  Mode CLI avec options:
    python lstm.py --cli --data ./data --output ./resultats --verbose

  Mode CLI silencieux:
    python lstm.py --cli --data ./data --output ./resultats --quiet
        """
    )
    
    parser.add_argument(
        '--cli', 
        action='store_true',
        help='Lance l\'application en mode ligne de commande (CLI) au lieu du mode GUI Streamlit'
    )
    
    cli_group = parser.add_argument_group('Options CLI')
    
    cli_group.add_argument(
        '--data', 
        type=str, 
        default='./data',
        help='Dossier contenant les données CSV des capteurs (défaut: ./data)'
    )
    
    cli_group.add_argument(
        '--output', 
        type=str, 
        default='./resultats',
        help='Dossier de sortie pour les modèles, graphiques et résultats (défaut: ./resultats)'
    )
    
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbeux: affiche plus d\'informations de débogage'
    )
    
    verbosity_group.add_argument(
        '--quiet', '-q',
        action='store_true', 
        help='Mode silencieux: affiche seulement les erreurs'
    )
    
    args = parser.parse_args()
    
    if args.cli:
        if not TABULATE_AVAILABLE:
            print("⚠️  Le module 'tabulate' n'est pas installé. Les tableaux seront affichés en format simple.")
            print("Pour une meilleure présentation, installez tabulate avec: pip install tabulate")
        
        run_cli_training(args)
        
    else:
        if not STREAMLIT_AVAILABLE:
            print("ERREUR: Streamlit n'est pas installé.", file=sys.stderr)
            print("Installez-le avec: pip install streamlit matplotlib", file=sys.stderr)
            print("Ou utilisez le mode CLI avec --cli", file=sys.stderr)
            sys.exit(1)
        
        main_streamlit()

if __name__ == "__main__":
    main()
