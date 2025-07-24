"""
Script: tuning_lstm.py
Auteur: Jules Lefèvre <jules.lefevre@etudiant.univ-reims.fr>
Date de création: 11/07/2025
Description: Script CLI pour l'optimisation automatique d'hyperparamètres de modèles LSTM 
            sur des données de flux de véhicules en utilisant Optuna. Le script utilise une 
            optimisation multi-objectifs (MAE% et complexité du modèle) avec validation croisée 
            temporelle, early stopping, et parallélisation sur GPU. Chaque capteur de chaque 
            intersection est traité indépendamment pour produire des modèles spécialisés.
            
Fonctionnalités:
- Optimisation bayésienne des hyperparamètres avec Optuna
- Validation croisée adaptée aux séries temporelles (TimeSeriesSplit)
- Parallélisation automatique sur plusieurs GPU
- Early stopping pour éviter le surapprentissage
- Gestion optimisée de la mémoire GPU
- Sauvegarde des modèles avec nomenclature descriptive
"""

import os
import re
import argparse
import random
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import TimeSeriesSplit
import optuna

import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION GLOBALE ET PARAMÈTRES PAR DÉFAUT
# =============================================================================

# Graine pour la reproductibilité des résultats
SEED = 42

# Dossier de sauvegarde des modèles entraînés
MODELS_DIR = "models"

# Seuil MAE en pourcentage pour privilégier la simplicité en cas d'égalité
THRESHOLD_MAE = 1.0

# Nombre maximum de workers pour les DataLoaders (limitation pour éviter "Too many open files")
MAX_WORKERS = 2

# Patience pour l'early stopping (nombre d'époques sans amélioration avant arrêt)
PATIENCE = 10

# =============================================================================
# FONCTIONS UTILITAIRES ET CONFIGURATION
# =============================================================================

def fixer_graine(seed=SEED):
    """
    Fixe toutes les graines pour assurer la reproductibilité des expériences.
    
    Configure les générateurs de nombres aléatoires de Python, NumPy et PyTorch
    pour obtenir des résultats déterministes à travers les exécutions.
    
    Args:
        seed (int): Valeur de la graine à utiliser
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Pour tous les GPU disponibles
    
    # Configuration CUDNN pour la reproductibilité (peut réduire les performances)
    cudnn.deterministic = True
    cudnn.benchmark = False

# =============================================================================
# FONCTIONS DE CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

def charger_donnees(dossier_racine):
    """
    Charge toutes les données de flux de véhicules depuis le dossier racine.
    
    Parcourt récursivement tous les dossiers d'intersections et charge les fichiers CSV
    contenant les données de capteurs. Effectue le nettoyage et la structuration des données.
    
    Args:
        dossier_racine (str): Chemin vers le dossier contenant les sous-dossiers d'intersections
        
    Returns:
        dict: Dictionnaire {nom_intersection: DataFrame_pivot} où chaque DataFrame
              a les timestamps en index et les capteurs en colonnes
    """
    donnees = {}
    logging.info(f"Chargement des données depuis: {dossier_racine}")
    
    # Vérification de l'existence du dossier racine
    if not os.path.exists(dossier_racine):
        logging.error(f"Dossier '{dossier_racine}' introuvable.")
        return donnees
    
    # Parcours de tous les dossiers d'intersections
    for inter in sorted(os.listdir(dossier_racine)):
        chemin = os.path.join(dossier_racine, inter)
        
        # Vérification que c'est bien un dossier
        if not os.path.isdir(chemin):
            continue
            
        # Recherche des fichiers CSV dans le dossier
        csvs = [f for f in os.listdir(chemin) if f.lower().endswith('.csv')]
        if not csvs:
            logging.warning(f"Aucun CSV dans '{inter}', skip.")
            continue
            
        try:
            # Chargement du premier fichier CSV trouvé avec colonnes spécifiques
            df = pd.read_csv(
                os.path.join(chemin, csvs[0]), 
                sep=';',  # Séparateur point-virgule
                encoding='utf-8',
                usecols=['count_point_name', 'measure_datetime', 'flow[veh/h]']  # Colonnes utiles seulement
            )
            
            # Conversion et nettoyage des timestamps
            df['measure_datetime'] = pd.to_datetime(
                df['measure_datetime'], utc=True, errors='coerce'
            )
            
            # Suppression des lignes avec des valeurs manquantes critiques
            df = df.dropna(subset=['measure_datetime', 'flow[veh/h]'])
            
            # Conversion du timezone UTC vers Europe/Paris (fuseau horaire français)
            df['measure_datetime'] = df['measure_datetime'].dt.tz_convert('Europe/Paris')
            
            # Création du tableau pivot: lignes=timestamps, colonnes=capteurs, valeurs=flux
            pivot = df.pivot(
                index='measure_datetime', 
                columns='count_point_name', 
                values='flow[veh/h]'
            ).sort_index()
            
            donnees[inter] = pivot
            logging.info(f"{inter}: {pivot.shape[0]} lignes, {pivot.shape[1]} capteurs")
            
        except Exception as e:
            logging.error(f"Erreur {inter}: {e}")
    
    logging.info(f"Total intersections chargées: {len(donnees)}")
    return donnees

def creer_caracteristiques(donnees):
    """
    Génère les caractéristiques d'entrée pour les modèles LSTM à partir des données brutes.
    
    Pour chaque capteur de chaque intersection, crée un ensemble de features:
    - flow: flux du capteur (variable cible)
    - mean_flow_others: flux moyen des autres capteurs de la même intersection
    - hour_cos: encodage cyclique de l'heure pour capturer la périodicité quotidienne
    - ma3, ma6, ma12: moyennes mobiles sur 3, 6 et 12 périodes pour les tendances
    
    Args:
        donnees (dict): Données pivot par intersection issues de charger_donnees()
        
    Returns:
        dict: {(intersection, capteur): DataFrame_features} où chaque DataFrame
              contient les 6 features nécessaires pour l'entraînement LSTM
    """
    feats = {}
    
    for inter, pivot in donnees.items():
        # Remplissage des valeurs manquantes par la moyenne de chaque capteur
        # (stratégie simple mais efficace pour les données de trafic)
        filled = pivot.fillna(pivot.mean())
        
        # Calcul du flux total de l'intersection à chaque timestamp
        total = filled.sum(axis=1).values.reshape(-1, 1)
        count = filled.shape[1]  # Nombre de capteurs
        
        # Calcul du flux moyen des "autres" capteurs pour chaque capteur
        # Formule: (total - capteur_actuel) / (nombre_capteurs - 1)
        # Ceci donne le contexte spatial pour chaque capteur
        others = (total - filled.values) / (count - 1)
        df_others = pd.DataFrame(others, index=filled.index, columns=filled.columns)
        
        # Traitement individuel de chaque capteur
        for cap in pivot.columns:
            # Création du DataFrame de caractéristiques pour ce capteur
            dfc = pd.DataFrame(index=pivot.index)
            
            # Feature 1: flux du capteur (variable cible à prédire)
            dfc['flow'] = pivot[cap]
            
            # Feature 2: contexte spatial (flux moyen des autres capteurs)
            dfc['mean_flow_others'] = df_others[cap]
            
            # Feature 3: contexte temporel cyclique (heure de la journée)
            # Utilisation du cosinus pour un encodage continu de la cyclicité
            heures = dfc.index.hour
            dfc['hour_cos'] = np.cos(2 * np.pi * heures / 24)
            
            # Features 4-6: moyennes mobiles pour capturer les tendances
            # Différentes fenêtres pour capturer des patterns à court/moyen terme
            for w in (3, 6, 12):
                dfc[f'ma{w}'] = dfc['flow'].rolling(window=w, min_periods=1).mean()
            
            # Suppression des lignes avec des valeurs manquantes dans la cible
            feats[(inter, cap)] = dfc.dropna(subset=['flow'])
    
    return feats

def creer_sequences(df, seq_len):
    """
    Transforme un DataFrame de caractéristiques en séquences temporelles pour LSTM.
    
    Crée des séquences glissantes de longueur seq_len où chaque séquence X
    est utilisée pour prédire la valeur y au timestep suivant.
    
    Args:
        df (DataFrame): DataFrame avec les 6 colonnes de caractéristiques
        seq_len (int): Longueur des séquences d'entrée (fenêtre temporelle)
        
    Returns:
        tuple: (X, y) où:
               - X.shape = (n_sequences, seq_len, 6_features)
               - y.shape = (n_sequences,)
    """
    # Ordre fixe des colonnes pour la cohérence entre entraînement et prédiction
    cols = ['flow', 'hour_cos', 'mean_flow_others', 'ma3', 'ma6', 'ma12']
    arr = df[cols].values
    
    X, y = [], []
    
    # Création des séquences glissantes
    for i in range(seq_len, len(arr)):
        # Séquence d'entrée: fenêtre de seq_len timesteps précédents
        X.append(arr[i-seq_len:i])
        # Valeur cible: flux au timestep actuel (index 0 = 'flow')
        y.append(arr[i, 0])
        
    return np.array(X), np.array(y)

# =============================================================================
# DÉFINITION DE L'ARCHITECTURE LSTM
# =============================================================================

class RegresseurLSTM(nn.Module):
    """
    Architecture LSTM pour la prédiction de flux de véhicules.
    
    Modèle séquentiel composé de:
    - Couches LSTM empilées avec dropout pour la régularisation
    - Couche linéaire finale pour la régression (sortie unique)
    
    Args:
        in_size (int): Nombre de features d'entrée (6 dans notre cas)
        hid_size (int): Taille des états cachés des couches LSTM
        n_layers (int): Nombre de couches LSTM empilées
        dropout (float): Taux de dropout entre les couches LSTM
    """
    def __init__(self, in_size, hid_size, n_layers, dropout):
        super().__init__()
        
        # Couches LSTM empilées avec dropout
        # batch_first=True pour faciliter la manipulation des tenseurs
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Couche de sortie pour la régression (1 valeur de flux)
        self.fc = nn.Linear(hid_size, 1)

    def forward(self, x):
        """
        Propagation avant du modèle.
        
        Args:
            x (torch.Tensor): Séquences d'entrée de forme (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: Prédictions de forme (batch_size, 1)
        """
        # Passage dans les couches LSTM
        # out: (batch_size, seq_len, hidden_size)
        # _: (h_n, c_n) états cachés finaux (non utilisés ici)
        out, _ = self.lstm(x)
        
        # Utilisation seulement du dernier timestep pour la prédiction
        # out[:, -1, :] extrait la sortie du dernier pas de temps
        return self.fc(out[:, -1, :])

# =============================================================================
# FONCTIONS D'ÉVALUATION ET D'ENTRAÎNEMENT
# =============================================================================

def evaluation_cv(df, params, device):
    """
    Évalue un jeu d'hyperparamètres via validation croisée temporelle.
    
    Utilise TimeSeriesSplit pour respecter l'ordre temporel des données,
    essentiel pour les séries temporelles. Implémente l'early stopping
    pour éviter le surapprentissage.
    
    Args:
        df (DataFrame): DataFrame avec les caractéristiques du capteur
        params (dict): Dictionnaire des hyperparamètres à tester
        device (torch.device): Device pour l'entraînement (CPU/GPU)
        
    Returns:
        float: MAE moyenne en pourcentage sur tous les folds de validation
    """
    # Création des séquences temporelles
    X, y = creer_sequences(df, params['window_size'])
    
    # Vérification de la taille minimale des données
    if X.size == 0:
        return np.inf
    
    # Flux moyen pour normaliser la MAE en pourcentage
    mean_flow = df['flow'].mean()
    
    # Configuration de la validation croisée temporelle
    # n_splits=5: 5 folds, gap=1: décalage pour éviter le data leakage
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    errors = []
    
    # Évaluation sur chaque fold
    for train_idx, test_idx in tscv.split(X):
        # Division des données pour ce fold
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        
        # Création des datasets PyTorch
        tr_ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)  # Ajout dimension pour compatibilité
        )
        te_ds = TensorDataset(
            torch.tensor(Xte, dtype=torch.float32),
            torch.tensor(yte, dtype=torch.float32).unsqueeze(1)
        )
        
        # DataLoaders avec configuration optimisée
        tr_ld = DataLoader(
            tr_ds, 
            batch_size=params['batch_size'], 
            shuffle=False,  # Pas de mélange pour préserver l'ordre temporel
            num_workers=MAX_WORKERS, 
            pin_memory=True  # Accélération transfert CPU->GPU
        )
        te_ld = DataLoader(
            te_ds, 
            batch_size=params['batch_size'], 
            shuffle=False,
            num_workers=MAX_WORKERS, 
            pin_memory=True
        )
        
        # Initialisation du modèle pour ce fold
        model = RegresseurLSTM(
            X.shape[2],  # Nombre de features (6)
            params['hidden_size'],
            params['num_layers'],
            params['dropout']
        ).to(device)
        
        # Configuration de l'optimiseur et de la fonction de perte
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.L1Loss()  # MAE (Mean Absolute Error)
        
        # Variables pour l'early stopping
        best_val = np.inf
        wait = 0
        
        # Boucle d'entraînement avec early stopping
        for epoch in range(params['num_epochs']):
            # Phase d'entraînement
            model.train()
            for xb, yb in tr_ld:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            
            # Phase de validation pour l'early stopping
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for xb, yb in te_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    val_preds.append(model(xb).cpu().numpy())
                    val_true.append(yb.cpu().numpy())
            
            # Calcul de l'erreur de validation
            val_err = np.abs(
                np.vstack(val_preds).flatten() - np.vstack(val_true).flatten()
            ).mean()
            
            # Logic d'early stopping
            if val_err < best_val:
                best_val = val_err
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break  # Arrêt anticipé si pas d'amélioration
        
        errors.append(best_val)
    
    # Retour de la MAE moyenne en pourcentage
    return 100.0 * np.mean(errors) / mean_flow

def train_final(df, params, path, device):
    """
    Entraîne le modèle final avec tous les paramètres optimaux sur toutes les données.
    
    Contrairement à evaluation_cv, cette fonction utilise toutes les données disponibles
    pour l'entraînement final du modèle qui sera sauvegardé.
    
    Args:
        df (DataFrame): DataFrame complet avec les caractéristiques
        params (dict): Hyperparamètres optimaux trouvés par Optuna
        path (str): Chemin de sauvegarde du modèle
        device (torch.device): Device pour l'entraînement
    """
    # Création des séquences sur toutes les données
    X, y = creer_sequences(df, params['window_size'])
    
    # Dataset et DataLoader
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    )
    ld = DataLoader(
        ds, 
        batch_size=params['batch_size'], 
        shuffle=False,
        num_workers=MAX_WORKERS, 
        pin_memory=True
    )
    
    # Initialisation du modèle final
    model = RegresseurLSTM(
        X.shape[2], 
        params['hidden_size'],
        params['num_layers'], 
        params['dropout']
    ).to(device)
    
    # Configuration de l'entraînement
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.L1Loss()
    
    # Entraînement complet sans early stopping
    for epoch in range(params['num_epochs']):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    # Sauvegarde du modèle entraîné
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)

# =============================================================================
# FONCTIONS UTILITAIRES POUR NOMMAGE ET ORGANISATION
# =============================================================================

def nom_fichier(inter, cap, params, mae_pct):
    """
    Génère un nom de fichier descriptif pour le modèle sauvegardé.
    
    Le nom encode tous les hyperparamètres et la performance du modèle
    pour faciliter l'identification et la comparaison des modèles.
    
    Format: sensor_CAPTEUR_INTERSECTION_bs{batch}_hs{hidden}_nl{layers}_
            do{dropout}_lr{learning_rate}_ep{epochs}_ws{window}_mae{mae}.pt
    
    Args:
        inter (str): Nom de l'intersection
        cap (str): Nom du capteur
        params (dict): Hyperparamètres du modèle
        mae_pct (float): MAE en pourcentage du modèle
        
    Returns:
        str: Nom de fichier formaté
    """
    # Nettoyage des noms pour éviter les caractères problématiques
    i = re.sub(r"\W+", "_", inter)  # Remplacement caractères non-alphanumériques
    c = re.sub(r"\W+", "_", str(cap))
    
    # Construction du nom avec tous les paramètres
    return (
        f"sensor_{c}_{i}_bs{params['batch_size']}_hs{params['hidden_size']}"
        f"_nl{params['num_layers']}_do{int(params['dropout']*100)}"
        f"_lr{params['learning_rate']:.0e}_ep{params['num_epochs']}"
        f"_ws{params['window_size']}_mae{int(round(mae_pct))}.pt"
    )

# =============================================================================
# FONCTION PRINCIPALE ET POINT D'ENTRÉE
# =============================================================================

def main():
    """
    Fonction principale orchestrant l'optimisation d'hyperparamètres.
    
    Processus complet:
    1. Configuration des arguments de ligne de commande
    2. Chargement et préparation des données
    3. Optimisation Optuna pour chaque capteur
    4. Entraînement et sauvegarde des modèles finaux
    """
    
    # Configuration du logging pour suivre le processus
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    # Configuration des arguments de ligne de commande
    parser = argparse.ArgumentParser("Tuning LSTM via Optuna (CLI)")
    parser.add_argument("--data", type=str, required=True,
                        help="Dossier racine des intersections")
    parser.add_argument("--trials", type=int, default=100,
                        help="Nombre d'essais Optuna par capteur")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_MAE,
                        help="Seuil MAE% pour privilégier simplicité")
    args = parser.parse_args()

    # Initialisation pour la reproductibilité
    fixer_graine()
    
    # Chargement et préparation des données
    donnees = charger_donnees(args.data)
    if not donnees:
        logging.error("Aucune donnée chargée. Arrêt.")
        return
    
    feats = creer_caracteristiques(donnees)
    logging.info(f"Total capteurs à traiter: {len(feats)}")
    
    # Création du dossier de sauvegarde
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Configuration Optuna
    storage = None
    pruner = optuna.pruners.MedianPruner()  # Arrêt précoce des trials non prometteurs

    # =========================================================================
    # BOUCLE PRINCIPALE: OPTIMISATION POUR CHAQUE CAPTEUR
    # =========================================================================
    
    for (inter, cap), df_current in feats.items():
        logging.info(f"=== {inter} / {cap} ===")
        
        # Vérification de la taille minimale des données
        if len(df_current) < 50:
            logging.warning(f"Skip {inter}/{cap}: moins de 50 échantillons")
            continue
        
        # Création de l'étude Optuna pour ce capteur
        study = optuna.create_study(
            study_name=f"study_{inter}_{cap}",
            directions=["minimize", "minimize"],  # Multi-objectif: MAE + complexité
            sampler=optuna.samplers.TPESampler(seed=SEED),  # Optimisation bayésienne
            pruner=pruner,
            storage=storage,
            load_if_exists=True  # Reprise d'étude existante si disponible
        )
        
        def objective(trial):
            """
            Fonction objectif pour l'optimisation Optuna.
            
            Définit l'espace de recherche des hyperparamètres et évalue
            chaque configuration selon deux critères: performance (MAE%) 
            et complexité (nombre de paramètres).
            
            Args:
                trial (optuna.Trial): Trial Optuna contenant les suggestions d'hyperparamètres
                
            Returns:
                tuple: (mae_percentage, nombre_paramètres) pour optimisation multi-objectif
            """
            # Définition de l'espace de recherche des hyperparamètres
            params = {
                'hidden_size':   trial.suggest_categorical('hidden_size', [64, 96, 128]),
                'num_layers':    trial.suggest_int('num_layers', 2, 4),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'dropout':       trial.suggest_float('dropout', 0.3, 0.5),
                'window_size':   trial.suggest_categorical('window_size', [24]),  # 24h de données
                'batch_size':    trial.suggest_categorical('batch_size', [16, 64, 128]),
                'num_epochs':    trial.suggest_int('num_epochs', 20, 40),
            }
            
            # Attribution du GPU pour la parallélisation
            ngpu = torch.cuda.device_count()
            if ngpu > 0:
                gpu_id = trial.number % ngpu  # Distribution cyclique sur les GPU
                device = torch.device(f'cuda:{gpu_id}')
            else:
                device = torch.device('cpu')
            
            try:
                # Évaluation des hyperparamètres via validation croisée
                mae = evaluation_cv(df_current, params, device)
                
                # Calcul du nombre de paramètres pour l'objectif de complexité
                model = RegresseurLSTM(6, params['hidden_size'], params['num_layers'], params['dropout'])
                param_count = sum(p.numel() for p in model.parameters())
                trial.set_user_attr('param_count', param_count)
                
                # Nettoyage mémoire GPU
                torch.cuda.empty_cache()
                
                return mae, param_count
                
            except RuntimeError as e:
                # Gestion des erreurs de mémoire GPU
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    return np.inf, np.inf
                raise

        # Lancement de l'optimisation
        ngpu = torch.cuda.device_count()
        n_jobs = ngpu if ngpu > 0 else 1  # Parallélisation selon le nombre de GPU
        
        study.optimize(objective, n_trials=args.trials, n_jobs=n_jobs)

        # Sélection du meilleur trial (meilleure MAE)
        best = min(study.best_trials, key=lambda t: t.values[0])
        params = best.params
        mae_pct = best.values[0]
        
        logging.info(f"Final: MAE%={mae_pct:.2f}, params={best.values[1]}")

        # Entraînement et sauvegarde du modèle final
        path = os.path.join(
            MODELS_DIR,
            nom_fichier(inter, cap, params, mae_pct)
        )
        
        train_final(
            df_current, 
            params, 
            path,
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        
        logging.info(f"Modèle enregistré: {path}")

    logging.info("=== FIN DU TUNING ===")

# =============================================================================
# POINT D'ENTRÉE DU SCRIPT
# =============================================================================

if __name__ == "__main__":
    """
    Point d'entrée principal du script.
    
    Exemple d'utilisation:
    python tuning_lstm.py --data ./data --trials 50
    """
    main()