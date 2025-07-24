"""
Script: analyse_similarite_lstm.py
Auteur: Jules Lefèvre <jules.lefevre@etudiant.univ-reims.fr>
Date de création: 21/07/2025
Description: Application Streamlit pour analyser la similarité entre modèles LSTM 
            entraînés sur différents capteurs de trafic. L'outil permet d'identifier 
            les modèles qui peuvent être consolidés en un seul modèle généraliste,
            basé sur l'analyse des performances croisées (test d'un modèle sur les 
            données d'un autre capteur). Inclut visualisations interactives, 
            clustering hiérarchique et recommandations de consolidation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import re
import glob
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DE L'APPLICATION STREAMLIT
# =============================================================================

# Configuration de la page Streamlit avec titre, icône et layout
st.set_page_config(
    page_title="Analyse de Similarité des Modèles LSTM",
    page_icon="🧠",
    layout="wide"
)

# =============================================================================
# DÉFINITION DE L'ARCHITECTURE LSTM
# =============================================================================

class RegresseurLSTM(nn.Module):
    """
    Classe définissant l'architecture du modèle LSTM pour la régression.
    
    Architecture:
    - Couches LSTM empilées avec dropout
    - Couche linéaire finale pour la prédiction
    
    Args:
        in_size (int): Nombre de features d'entrée (6 dans notre cas)
        hid_size (int): Taille des couches cachées LSTM
        n_layers (int): Nombre de couches LSTM empilées
        dropout (float): Taux de dropout entre les couches LSTM
    """
    def __init__(self, in_size, hid_size, n_layers, dropout):
        super().__init__()
        # Couches LSTM empilées avec dropout et batch_first=True pour faciliter le traitement
        self.lstm = nn.LSTM(in_size, hid_size, n_layers,
                            dropout=dropout, batch_first=True)
        # Couche linéaire finale pour prédire une seule valeur (flux de véhicules)
        self.fc = nn.Linear(hid_size, 1)

    def forward(self, x):
        """
        Propagation avant du modèle.
        
        Args:
            x (torch.Tensor): Séquences d'entrée de forme (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: Prédictions de forme (batch_size, 1)
        """
        # Passage dans les couches LSTM, on récupère seulement les outputs
        out, _ = self.lstm(x)
        # On utilise seulement le dernier timestep pour la prédiction
        return self.fc(out[:, -1, :])

# =============================================================================
# FONCTIONS DE CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

@st.cache_data
def charger_donnees_selectif(dossier_racine, intersections_necessaires):
    """
    Charge sélectivement les données CSV des intersections nécessaires pour optimiser les performances.
    
    Cette fonction lit uniquement les données des intersections qui ont des modèles associés,
    évitant ainsi de charger inutilement toutes les données disponibles.
    
    Args:
        dossier_racine (str): Chemin vers le dossier contenant les sous-dossiers d'intersections
        intersections_necessaires (set): Ensemble des noms d'intersections à charger
        
    Returns:
        dict: Dictionnaire {nom_intersection: DataFrame_pivot} avec les données temporelles
              indexées par datetime et colonnes = capteurs
    """
    donnees = {}
    
    # Vérification de l'existence du dossier racine
    if not os.path.exists(dossier_racine):
        st.error(f"Dossier '{dossier_racine}' introuvable.")
        return donnees
    
    st.info(f"🎯 Chargement sélectif pour: {list(intersections_necessaires)}")
    
    # Parcours de tous les dossiers d'intersections
    for inter in sorted(os.listdir(dossier_racine)):
        # Vérifier si cette intersection est nécessaire (optimisation performance)
        if not any(intersection in inter for intersection in intersections_necessaires):
            continue
            
        chemin = os.path.join(dossier_racine, inter)
        if not os.path.isdir(chemin):
            continue
            
        # Recherche du fichier CSV dans le dossier
        csvs = [f for f in os.listdir(chemin) if f.lower().endswith('.csv')]
        if not csvs:
            continue
            
        try:
            # Chargement optimisé: seulement les colonnes nécessaires
            df = pd.read_csv(
                os.path.join(chemin, csvs[0]), sep=';', encoding='utf-8',
                usecols=['count_point_name', 'measure_datetime', 'flow[veh/h]']
            )
            
            # Conversion des dates avec gestion des erreurs
            df['measure_datetime'] = pd.to_datetime(
                df['measure_datetime'], utc=True, errors='coerce'
            )
            
            # Suppression des lignes avec des valeurs manquantes critiques
            df = df.dropna(subset=['measure_datetime', 'flow[veh/h]'])
            
            # Conversion du timezone UTC vers Europe/Paris
            df['measure_datetime'] = df['measure_datetime'].dt.tz_convert('Europe/Paris')
            
            # Création du tableau pivot: lignes=datetime, colonnes=capteurs, valeurs=flux
            pivot = df.pivot(
                index='measure_datetime', columns='count_point_name', values='flow[veh/h]'
            ).sort_index()
            
            donnees[inter] = pivot
            st.success(f"✅ {inter}: {pivot.shape[0]} lignes, {pivot.shape[1]} capteurs")
            
        except Exception as e:
            st.error(f"❌ Erreur {inter}: {e}")
    
    return donnees

@st.cache_data
def creer_caracteristiques_selectif(donnees, capteurs_necessaires):
    """
    Crée les caractéristiques d'entrée pour les modèles LSTM, uniquement pour les capteurs nécessaires.
    
    Pour chaque capteur, génère:
    - flow: flux du capteur (variable cible)
    - mean_flow_others: flux moyen des autres capteurs de la même intersection
    - hour_cos: encodage cyclique de l'heure (cosinus)
    - ma3, ma6, ma12: moyennes mobiles sur 3, 6 et 12 périodes
    
    Args:
        donnees (dict): Données pivot par intersection
        capteurs_necessaires (dict): {intersection: set_of_capteurs} à traiter
        
    Returns:
        dict: {(intersection, capteur): DataFrame_avec_features}
    """
    feats = {}
    
    for inter, pivot in donnees.items():
        # Identifier quels capteurs sont nécessaires pour cette intersection
        capteurs_inter = set()
        for intersection, capteurs in capteurs_necessaires.items():
            if intersection in inter:
                capteurs_inter.update(capteurs)
        
        if not capteurs_inter:
            continue
            
        # Remplissage des valeurs manquantes par la moyenne de chaque capteur
        filled = pivot.fillna(pivot.mean())
        
        # Calcul du flux total de l'intersection et du nombre de capteurs
        total = filled.sum(axis=1).values.reshape(-1, 1)
        count = filled.shape[1]
        
        # Calcul du flux moyen des "autres" capteurs pour chaque capteur
        # (total - capteur_actuel) / (nombre_capteurs - 1)
        others = (total - filled.values) / (count - 1)
        df_others = pd.DataFrame(others, index=filled.index, columns=filled.columns)
        
        # Traitement de chaque capteur individuellement
        for cap in pivot.columns:
            # Vérifier si ce capteur est nécessaire (optimisation)
            if str(cap) not in capteurs_inter:
                continue
                
            # Création du DataFrame de caractéristiques pour ce capteur
            dfc = pd.DataFrame(index=pivot.index)
            
            # Feature 1: flux du capteur (variable cible)
            dfc['flow'] = pivot[cap]
            
            # Feature 2: flux moyen des autres capteurs (contexte spatial)
            dfc['mean_flow_others'] = df_others[cap]
            
            # Feature 3: encodage cyclique de l'heure (contexte temporel)
            heures = dfc.index.hour
            dfc['hour_cos'] = np.cos(2 * np.pi * heures / 24)
            
            # Features 4-6: moyennes mobiles pour capturer les tendances
            for w in (3, 6, 12):
                dfc[f'ma{w}'] = dfc['flow'].rolling(window=w, min_periods=1).mean()
            
            # Suppression des lignes avec des valeurs manquantes dans 'flow'
            feats[(inter, cap)] = dfc.dropna(subset=['flow'])
            st.info(f"📊 Caractéristiques créées pour {inter} - capteur {cap}")
    
    return feats

# =============================================================================
# FONCTIONS DE PRÉPARATION DES SÉQUENCES ET PARSING
# =============================================================================

def creer_sequences(df, seq_len):
    """
    Transforme un DataFrame de caractéristiques en séquences pour l'entraînement LSTM.
    
    Crée des séquences glissantes de longueur seq_len, où chaque séquence X prédit
    la valeur y au timestep suivant.
    
    Args:
        df (DataFrame): DataFrame avec les colonnes de caractéristiques
        seq_len (int): Longueur des séquences d'entrée
        
    Returns:
        tuple: (X, y) où X.shape = (n_sequences, seq_len, n_features)
               et y.shape = (n_sequences,)
    """
    # Colonnes de caractéristiques dans l'ordre attendu par le modèle
    cols = ['flow', 'hour_cos', 'mean_flow_others', 'ma3', 'ma6', 'ma12']
    arr = df[cols].values
    
    X, y = [], []
    # Création des séquences glissantes
    for i in range(seq_len, len(arr)):
        # Séquence d'entrée: timesteps [i-seq_len:i]
        X.append(arr[i-seq_len:i])
        # Valeur cible: flux au timestep i
        y.append(arr[i, 0])  # 0 = index de 'flow'
        
    return np.array(X), np.array(y)

def parser_nom_modele(nom_fichier):
    """
    Parse le nom d'un fichier de modèle pour extraire ses hyperparamètres.
    
    Format attendu: sensor_CAPTEUR_INTERSECTION_bs_hs_nl_do_lr_ep_ws_mae.pt
    Exemple: sensor_A12_Intersection1_bs32_hs64_nl2_do20_lr1e-3_ep100_ws24_mae150.pt
    
    Args:
        nom_fichier (str): Nom du fichier de modèle
        
    Returns:
        dict or None: Dictionnaire des paramètres extraits ou None si parsing échoue
    """
    # Expression régulière pour extraire tous les paramètres du nom de fichier
    pattern = r'sensor_(.+)_bs(\d+)_hs(\d+)_nl(\d+)_do(\d+)_lr(\d+e-\d+)_ep(\d+)_ws(\d+)_mae(\d+)\.pt'
    match = re.match(pattern, nom_fichier)
    
    if match:
        return {
            'capteur_inter': match.group(1),      # Identifiant capteur_intersection
            'batch_size': int(match.group(2)),    # Taille des batchs
            'hidden_size': int(match.group(3)),   # Taille des couches cachées
            'num_layers': int(match.group(4)),    # Nombre de couches LSTM
            'dropout': int(match.group(5)) / 100.0,  # Taux de dropout (converti en decimal)
            'learning_rate': float(match.group(6)),  # Taux d'apprentissage
            'num_epochs': int(match.group(7)),    # Nombre d'époques d'entraînement
            'window_size': int(match.group(8)),   # Taille des séquences
            'mae_original': int(match.group(9))   # MAE original du modèle (en centièmes de %)
        }
    return None

def extraire_capteur_intersection(capteur_inter):
    """
    Sépare l'identifiant capteur_intersection en capteur et intersection.
    
    Args:
        capteur_inter (str): Chaîne au format "capteur_intersection"
        
    Returns:
        tuple: (capteur, intersection)
    """
    parts = capteur_inter.split('_')
    if len(parts) >= 2:
        capteur = parts[0]  # Premier élément = nom du capteur
        intersection = '_'.join(parts[1:])  # Reste = nom de l'intersection
        return capteur, intersection
    return capteur_inter, ""

# =============================================================================
# FONCTIONS DE CHARGEMENT ET ÉVALUATION DES MODÈLES
# =============================================================================

@st.cache_resource
def charger_modele(chemin_modele, params, device):
    """
    Charge un modèle LSTM depuis un fichier .pt.
    
    Utilise le décorateur @st.cache_resource pour éviter de recharger
    le même modèle plusieurs fois pendant la session.
    
    Args:
        chemin_modele (str): Chemin vers le fichier .pt
        params (dict): Paramètres du modèle pour reconstruire l'architecture
        device (torch.device): Device sur lequel charger le modèle
        
    Returns:
        RegresseurLSTM: Modèle chargé et prêt pour l'évaluation
    """
    # Reconstruction de l'architecture avec les paramètres extraits
    model = RegresseurLSTM(6, params['hidden_size'], params['num_layers'], params['dropout'])
    
    # Chargement des poids sauvegardés
    model.load_state_dict(torch.load(chemin_modele, map_location=device))
    
    # Déplacement vers le device approprié et passage en mode évaluation
    model.to(device)
    model.eval()
    
    return model

def evaluer_modele(model, X_test, y_test, device, mean_flow):
    """
    Évalue un modèle sur un jeu de données de test.
    
    Calcule la MAE en pourcentage par rapport au flux moyen pour une interprétation
    plus intuitive des performances.
    
    Args:
        model (RegresseurLSTM): Modèle à évaluer
        X_test (np.array): Séquences de test
        y_test (np.array): Valeurs cibles de test
        device (torch.device): Device pour les calculs
        mean_flow (float): Flux moyen pour normaliser la MAE
        
    Returns:
        tuple: (mae_percentage, predictions)
    """
    model.eval()
    with torch.no_grad():
        # Conversion en tenseurs PyTorch
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # Traitement par batch pour éviter les problèmes de mémoire
        batch_size = 256
        predictions = []
        
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_pred = model(batch_X)
            predictions.append(batch_pred.cpu().numpy())
        
        # Concaténation de tous les batchs
        predictions = np.concatenate(predictions, axis=0).flatten()
        
        # Calcul de la MAE absolue et en pourcentage
        mae_abs = np.abs(predictions - y_test).mean()
        mae_pct = 100.0 * mae_abs / mean_flow
        
    return mae_pct, predictions

# =============================================================================
# FONCTION PRINCIPALE DE L'APPLICATION
# =============================================================================

def main():
    """
    Fonction principale de l'application Streamlit.
    
    Orchestration complète de l'analyse de similarité:
    1. Configuration de l'interface utilisateur
    2. Chargement sélectif des données et modèles
    3. Calcul de la matrice de performance croisée
    4. Analyse de similarité et clustering
    5. Génération des recommandations
    """
    
    # =============================================================================
    # INTERFACE UTILISATEUR ET CONFIGURATION
    # =============================================================================
    
    st.title("🧠 Analyse de Similarité des Modèles LSTM")
    st.markdown("---")
    
    # Sidebar pour la configuration
    st.sidebar.header("Configuration")
    
    # Sélection des dossiers de données et modèles
    data_folder = st.sidebar.text_input("Dossier des données", value="./data")
    models_folder = st.sidebar.text_input("Dossier des modèles", value="./models")
    
    # Seuil de similarité pour regrouper les modèles
    seuil_similarite = st.sidebar.slider("Seuil de différence acceptable (MAE%)", 
                                        min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    
    # Détection automatique du device (GPU si disponible, sinon CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"Device utilisé: {device}")
    
    # Vérification de l'existence des dossiers
    if not os.path.exists(data_folder) or not os.path.exists(models_folder):
        st.error("Veuillez vérifier les chemins des dossiers de données et de modèles.")
        return
    
    # =============================================================================
    # DÉCOUVERTE ET PARSING DES MODÈLES DISPONIBLES
    # =============================================================================
    
    # Chargement de la liste des modèles disponibles
    modeles_disponibles = glob.glob(os.path.join(models_folder, "*.pt"))
    if not modeles_disponibles:
        st.error("Aucun modèle trouvé dans le dossier spécifié.")
        return
    
    st.info(f"🎯 Modèles trouvés: {len(modeles_disponibles)}")
    
    # Parsing des informations des modèles et identification des données nécessaires
    info_modeles = {}
    intersections_necessaires = set()  # Intersections à charger
    capteurs_necessaires = {}  # Capteurs à traiter par intersection
    
    for chemin in modeles_disponibles:
        nom = os.path.basename(chemin)
        params = parser_nom_modele(nom)
        
        if params:
            # Extraction du capteur et de l'intersection depuis le nom
            capteur, intersection = extraire_capteur_intersection(params['capteur_inter'])
            
            # Stockage des informations du modèle
            info_modeles[nom] = {
                'chemin': chemin,
                'params': params,
                'capteur': capteur,
                'intersection': intersection,
                'cle_feat': None  # Sera rempli lors du matching avec les données
            }
            
            # Ajout aux ensembles de données nécessaires
            intersections_necessaires.add(intersection)
            if intersection not in capteurs_necessaires:
                capteurs_necessaires[intersection] = set()
            capteurs_necessaires[intersection].add(capteur)
    
    # Affichage des intersections et capteurs identifiés
    st.write(f"**Intersections nécessaires**: {list(intersections_necessaires)}")
    for inter, caps in capteurs_necessaires.items():
        st.write(f"- {inter}: capteurs {list(caps)}")
    
    # =============================================================================
    # CHARGEMENT SÉLECTIF DES DONNÉES
    # =============================================================================
    
    # Chargement optimisé: uniquement les intersections et capteurs nécessaires
    with st.spinner("Chargement sélectif des données..."):
        donnees = charger_donnees_selectif(data_folder, intersections_necessaires)
        if not donnees:
            st.error("Aucune donnée chargée.")
            return
        
        # Création des caractéristiques pour les modèles LSTM
        feats = creer_caracteristiques_selectif(donnees, capteurs_necessaires)
        st.success(f"Données chargées: {len(feats)} capteurs trouvés")
    
    # =============================================================================
    # MATCHING MODÈLES-DONNÉES
    # =============================================================================
    
    # Association de chaque modèle avec ses données correspondantes
    for nom, info in info_modeles.items():
        intersection = info['intersection']
        capteur = info['capteur']
        
        # Recherche de la clé correspondante dans les caractéristiques chargées
        for (inter, cap), _ in feats.items():
            if intersection in inter and capteur == str(cap):
                info_modeles[nom]['cle_feat'] = (inter, cap)
                break
    
    # Filtrage des modèles pour lesquels les données sont disponibles
    modeles_valides = {k: v for k, v in info_modeles.items() if v['cle_feat'] is not None}
    
    if not modeles_valides:
        st.error("Aucun modèle ne correspond aux données disponibles.")
        return
    
    st.info(f"Modèles valides trouvés: {len(modeles_valides)}")
    
    # =============================================================================
    # SÉLECTION DES MODÈLES À ANALYSER
    # =============================================================================
    
    st.header("Sélection des Modèles à Analyser")
    
    noms_modeles = list(modeles_valides.keys())
    modeles_selectionnes = st.multiselect(
        "Sélectionnez les modèles à comparer:",
        options=noms_modeles,
        default=noms_modeles[:min(5, len(noms_modeles))],  # Max 5 par défaut
        help="Sélectionnez au moins 2 modèles pour l'analyse de similarité"
    )
    
    if len(modeles_selectionnes) < 2:
        st.warning("Veuillez sélectionner au moins 2 modèles pour l'analyse.")
        return
    
    # =============================================================================
    # ANALYSE DE SIMILARITÉ PRINCIPALE
    # =============================================================================
    
    if st.button("🔍 Lancer l'Analyse de Similarité"):
        
        # =========================================================================
        # PRÉPARATION DES DONNÉES DE TEST
        # =========================================================================
        
        donnees_test = {}
        st.write("🔧 Préparation des données de test...")
        
        # Pour chaque modèle sélectionné, préparer ses données de test
        for nom_modele in modeles_selectionnes:
            cle_feat = modeles_valides[nom_modele]['cle_feat']
            if cle_feat not in feats:
                st.error(f"Clé {cle_feat} introuvable dans feats")
                continue
                
            df = feats[cle_feat]
            st.write(f"📊 {nom_modele.split('_')[1]}: {len(df)} échantillons totaux")
            
            # Division 80-20 pour train/test (on utilise seulement le test)
            split_idx = int(0.8 * len(df))
            df_test = df.iloc[split_idx:]
            st.write(f"Test: {len(df_test)} échantillons")
            
            # Création des séquences pour ce modèle
            params = modeles_valides[nom_modele]['params']
            X_test, y_test = creer_sequences(df_test, params['window_size'])
            
            st.write(f"Séquences: {X_test.shape if len(X_test) > 0 else 'Aucune'}")
            
            # Stockage des données de test si des séquences ont été créées
            if len(X_test) > 0:
                donnees_test[nom_modele] = {
                    'X_test': X_test,
                    'y_test': y_test,
                    'mean_flow': df['flow'].mean(),  # Pour normaliser la MAE
                    'params': params
                }
                st.success(f"✅ Données test préparées pour {nom_modele.split('_')[1]}")
            else:
                st.error(f"❌ Aucune séquence générée pour {nom_modele.split('_')[1]}")
        
        st.write(f"Total modèles avec données de test: {len(donnees_test)}")
        
        if len(donnees_test) < 2:
            st.error("Pas assez de données de test valides pour l'analyse")
            return
        
        # =========================================================================
        # CALCUL DE LA MATRICE DE PERFORMANCE CROISÉE
        # =========================================================================
        
        st.header("📊 Matrice de Similarité")
        
        n_modeles = len(modeles_selectionnes)
        matrice_mae = np.zeros((n_modeles, n_modeles))
        
        # Barres de progression pour l'utilisateur
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calcul de toutes les combinaisons modèle_A testé sur données_B
        for i, modele_a in enumerate(modeles_selectionnes):
            for j, modele_b in enumerate(modeles_selectionnes):
                
                # Mise à jour de la progression
                progress = (i * n_modeles + j + 1) / (n_modeles * n_modeles)
                progress_bar.progress(progress)
                status_text.text(f"Test: {modele_a.split('_')[1]} sur données de {modele_b.split('_')[1]}")
                
                if i == j:
                    # Performance native du modèle (diagonale de la matrice)
                    matrice_mae[i, j] = modeles_valides[modele_a]['params']['mae_original']
                else:
                    # Test croisé: modèle A testé sur données B
                    try:
                        # Vérification de la disponibilité des données de test
                        if modele_b not in donnees_test:
                            matrice_mae[i, j] = np.inf
                            continue
                            
                        data_b = donnees_test[modele_b]
                        
                        # Chargement du modèle A
                        model = charger_modele(
                            modeles_valides[modele_a]['chemin'],
                            modeles_valides[modele_a]['params'],
                            device
                        )
                        
                        # Évaluation du modèle A sur les données B
                        mae_pct, predictions = evaluer_modele(
                            model, data_b['X_test'], data_b['y_test'], 
                            device, data_b['mean_flow']
                        )
                        
                        matrice_mae[i, j] = mae_pct
                        
                        # Nettoyage mémoire pour éviter l'accumulation
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors du test de {modele_a.split('_')[1]} sur {modele_b.split('_')[1]}: {e}")
                        matrice_mae[i, j] = np.inf
        
        # Nettoyage de l'interface
        progress_bar.empty()
        status_text.empty()
        
        # =========================================================================
        # AFFICHAGE DES RÉSULTATS DE PERFORMANCE
        # =========================================================================
        
        st.success("🎯 Analyse de similarité terminée !")
        
        # Tableau récapitulatif des performances croisées
        st.subheader("📋 Résumé des Performances Croisées")
        
        resultats_croises = []
        for i, modele_a in enumerate(modeles_selectionnes):
            for j, modele_b in enumerate(modeles_selectionnes):
                capteur_a = modeles_valides[modele_a]['capteur']
                capteur_b = modeles_valides[modele_b]['capteur']
                mae_value = matrice_mae[i, j]
                
                if i == j:
                    # Performance native
                    status = "🏠 Natif"
                    diff = 0.0
                else:
                    # Performance croisée vs native
                    mae_native = matrice_mae[j, j]  # Performance native du modèle B
                    diff = mae_value - mae_native
                    
                    # Classification de la performance selon la différence avec le modèle natif
                    if diff <= 5.0:
                        status = "✅ Excellent" if diff <= 2.0 else "🟢 Bon"
                    elif diff <= 10.0:
                        status = "🟡 Moyen"
                    else:
                        status = "🔴 Faible"
                
                # Ajout des résultats au tableau récapitulatif
                resultats_croises.append({
                    'Modèle': f"Capteur {capteur_a}",
                    'Testé sur': f"Capteur {capteur_b}",
                    'MAE (%)': f"{mae_value:.2f}",
                    'Différence': f"{diff:+.2f}%" if i != j else "0.00%",
                    'Status': status
                })
        
        # Affichage du tableau sous forme de DataFrame interactif
        df_resultats = pd.DataFrame(resultats_croises)
        st.dataframe(df_resultats, use_container_width=True, hide_index=True)
        
        # =========================================================================
        # VISUALISATION DE LA MATRICE DE PERFORMANCE
        # =========================================================================
        
        # Création d'une heatmap interactive avec Plotly
        df_matrice = pd.DataFrame(
            matrice_mae,
            index=[f"Capteur_{modeles_valides[modeles_selectionnes[i]]['capteur']}" for i in range(n_modeles)],
            columns=[f"Données_{modeles_valides[modeles_selectionnes[i]]['capteur']}" for i in range(n_modeles)]
        )
        
        # Remplacement des valeurs infinies pour l'affichage
        df_matrice_display = df_matrice.replace([np.inf, -np.inf], 999)
        
        # Création de la heatmap avec échelle de couleurs appropriée
        fig_heatmap = px.imshow(
            df_matrice_display,
            labels=dict(x="Données de test", y="Modèles", color="MAE %"),
            title="Matrice de Performance Croisée (MAE %)",
            color_continuous_scale="RdYlBu_r",  # Rouge = mauvais, Bleu = bon
            text_auto=".1f"  # Affichage des valeurs avec 1 décimale
        )
        fig_heatmap.update_layout(height=500)
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # =========================================================================
        # ANALYSE DE SIMILARITÉ ET CLUSTERING
        # =========================================================================
        
        st.header("🔗 Analyse de Regroupement")
        
        # Calcul des différences relatives entre performances croisées et natives
        matrice_diff = np.zeros((n_modeles, n_modeles))
        for i in range(n_modeles):
            for j in range(n_modeles):
                if i != j:
                    mae_native = matrice_mae[j, j]  # Performance native du modèle j
                    mae_croisee = matrice_mae[i, j]  # Performance du modèle i sur données j
                    diff_relative = abs(mae_croisee - mae_native)
                    matrice_diff[i, j] = diff_relative
        
        # Identification des paires de modèles similaires
        modeles_similaires = []
        for i in range(n_modeles):
            for j in range(i + 1, n_modeles):
                # Différence bidirectionnelle (i sur j ET j sur i)
                diff_ij = matrice_diff[i, j]  # Modèle i testé sur données j
                diff_ji = matrice_diff[j, i]  # Modèle j testé sur données i
                diff_moyenne = (diff_ij + diff_ji) / 2
                
                # Test du seuil de similarité
                if diff_moyenne <= seuil_similarite:
                    modeles_similaires.append((i, j, diff_moyenne))
        
        # =========================================================================
        # AFFICHAGE DES MODÈLES SIMILAIRES
        # =========================================================================
        
        if modeles_similaires:
            st.success(f"🎯 Modèles similaires trouvés (seuil: {seuil_similarite}% MAE):")
            
            cols = st.columns(2)
            
            # Colonne 1: Liste des paires similaires
            with cols[0]:
                for i, j, diff in modeles_similaires:
                    nom_i = modeles_selectionnes[i]
                    nom_j = modeles_selectionnes[j]
                    
                    st.info(f"**{modeles_valides[nom_i]['capteur']} ({modeles_valides[nom_i]['intersection']})** ↔ "
                           f"**{modeles_valides[nom_j]['capteur']} ({modeles_valides[nom_j]['intersection']})**\n\n"
                           f"Différence moyenne: {diff:.2f}% MAE")
            
            # Colonne 2: Dendrogramme de clustering hiérarchique
            with cols[1]:
                st.subheader("Dendrogramme de Regroupement")
                
                # Construction de la matrice de distance symétrique
                distance_matrix = np.zeros((n_modeles, n_modeles))
                for i in range(n_modeles):
                    for j in range(n_modeles):
                        if i != j:
                            diff_ij = matrice_diff[i, j]
                            diff_ji = matrice_diff[j, i]
                            distance_matrix[i, j] = (diff_ij + diff_ji) / 2
                
                # Clustering hiérarchique avec méthode de Ward
                linkage_matrix = linkage(squareform(distance_matrix), method='ward')
                
                # Création du dendrogramme
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = [f"{modeles_valides[modeles_selectionnes[i]]['capteur']}" for i in range(n_modeles)]
                dendrogram(linkage_matrix, labels=labels, ax=ax, orientation='top')
                ax.set_title("Dendrogramme de Similarité des Modèles")
                ax.set_ylabel("Distance")
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        else:
            st.warning(f"Aucun modèle similaire trouvé avec le seuil de {seuil_similarite}% MAE.")
        
        # =========================================================================
        # GÉNÉRATION DES RECOMMANDATIONS
        # =========================================================================
        
        st.header("💡 Recommandations")
        
        if modeles_similaires:
            st.markdown("### Stratégies de Consolidation Possibles:")
            
            # Algorithme de regroupement des modèles similaires
            groupes = []
            modeles_assignes = set()
            
            # Première passe: créer des groupes initiaux
            for i, j, diff in modeles_similaires:
                if i not in modeles_assignes and j not in modeles_assignes:
                    groupes.append([i, j])
                    modeles_assignes.update([i, j])
                elif i in modeles_assignes:
                    # Ajouter j au groupe contenant i
                    for groupe in groupes:
                        if i in groupe and j not in modeles_assignes:
                            groupe.append(j)
                            modeles_assignes.add(j)
                            break
                elif j in modeles_assignes:
                    # Ajouter i au groupe contenant j
                    for groupe in groupes:
                        if j in groupe and i not in modeles_assignes:
                            groupe.append(i)
                            modeles_assignes.add(i)
                            break
            
            # Affichage des groupes identifiés
            for idx, groupe in enumerate(groupes, 1):
                st.success(f"**Groupe {idx}**: Peut utiliser un modèle commun")
                for model_idx in groupe:
                    nom = modeles_selectionnes[model_idx]
                    capteur = modeles_valides[nom]['capteur']
                    intersection = modeles_valides[nom]['intersection']
                    mae_orig = modeles_valides[nom]['params']['mae_original']
                    st.write(f"- {capteur} ({intersection}) - MAE original: {mae_orig}%")
            
            # Calcul des métriques de réduction
            reduction_modeles = len(modeles_similaires)
            reduction_pct = (reduction_modeles / len(modeles_selectionnes)) * 100
            
            st.info(f"**Réduction potentielle**: {reduction_modeles} modèles en moins "
                   f"({reduction_pct:.1f}% de réduction)")
        
        else:
            st.markdown("### Recommandations:")
            st.write("- Tous les modèles semblent spécifiques à leur capteur")
            st.write("- Considérez d'assouplir le seuil de similarité si approprié")
            st.write("- Analysez les caractéristiques des capteurs pour identifier des groupes logiques")
        
        # =========================================================================
        # TABLEAU DÉTAILLÉ DES RÉSULTATS
        # =========================================================================
        
        st.header("📋 Résultats Détaillés")
        
        # Compilation des informations détaillées de chaque modèle
        resultats_details = []
        for i, nom_modele in enumerate(modeles_selectionnes):
            info = modeles_valides[nom_modele]
            resultats_details.append({
                'Modèle': f"Modèle_{i+1}",
                'Capteur': info['capteur'],
                'Intersection': info['intersection'],
                'MAE Original (%)': info['params']['mae_original'],
                'Hidden Size': info['params']['hidden_size'],
                'Layers': info['params']['num_layers'],
                'Dropout': info['params']['dropout'],
                'Window Size': info['params']['window_size']
            })
        
        # Affichage du tableau récapitulatif
        df_details = pd.DataFrame(resultats_details)
        st.dataframe(df_details, use_container_width=True)

# =============================================================================
# POINT D'ENTRÉE DE L'APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
