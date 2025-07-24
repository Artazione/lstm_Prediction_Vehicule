# 🚦 Prédiction de Flux Véhicules avec Modèles LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Projet réalisé dans le cadre de mon stage de M1 Informatique Parcours Intelligence Artificielle**  
> **Laboratoire CReSTIC - Lab-i*** - Université de Reims Champagne-Ardennes  
> **Période :** Juillet 2025

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Structure du Projet](#-structure-du-projet)
- [Guide d'Utilisation](#-guide-dutilisation)
- [Exemples Détaillés](#-exemples-détaillés)
- [Architecture LSTM](#-architecture-lstm)
- [Datasets](#-datasets)
- [Remerciements](#-remerciements)
- [Licence](#-licence)

## 🎯 À Propos

Ce projet développe une solution complète de **prédiction de flux de véhicules** utilisant des modèles **LSTM (Long Short-Term Memory)** pour analyser et prévoir le trafic routier. Le système comprend quatre composants principaux permettant l'optimisation d'hyperparamètres, l'entraînement de modèles, l'évaluation des performances et l'analyse de similarité entre modèles.

### Contexte Académique

Ce travail a été réalisé dans le cadre de mon **stage de M1 Informatique Parcours Intelligence Artificielle** au sein du **Laboratoire CReSTIC - Lab-i*** de l'**Université de Reims Champagne-Ardennes**.

Le projet s'inscrit dans une démarche de recherche appliquée sur l'utilisation de l'intelligence artificielle pour l'optimisation du trafic urbain et la prédiction des flux de véhicules en temps réel.

## ✨ Fonctionnalités

### 🚀 **Launcher Unifié (`main.py`)**
- Interface interactive pour tous les composants
- Détection automatique des dépendances
- Lancement direct en ligne de commande
- Diagnostic complet du système

### 🔧 **Optimisation d'Hyperparamètres (`tuning_lstm.py`)**
- Optimisation bayésienne avec **Optuna**
- Validation croisée temporelle (TimeSeriesSplit)
- Parallélisation multi-GPU automatique
- Early stopping intelligent
- Sauvegarde automatique des modèles optimaux

### 🧠 **Entraînement et Évaluation (`lstm.py`)**
- **Interface GUI** (Streamlit) et **CLI**
- Entraînement de modèles LSTM personnalisés
- Validation croisée optionnelle
- Visualisations interactives des prédictions
- Support multi-capteurs

### 📊 **Analyse de Similarité (`analyse_similarite_lstm.py`)**
- Analyse de performance croisée entre modèles
- Clustering hiérarchique des modèles similaires
- Recommandations de consolidation
- Matrices de similarité interactives
- Interface Streamlit dédiée

## 🛠 Installation

### Prérequis
- Python 3.8 ou supérieur
- CUDA (optionnel, pour l'accélération GPU)

### Installation des Dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-username/lstm-traffic-prediction.git
cd lstm-traffic-prediction

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales

Le fichier `requirements.txt` contient toutes les dépendances nécessaires :

```
# Core ML libraries
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0

# Optimization
optuna>=3.0.0

# GUI and Visualization
streamlit>=1.28.0
plotly>=5.0.0
matplotlib>=3.5.0

# Data processing
scipy>=1.7.0
python-dateutil>=2.8.0

# CLI utilities
tabulate>=0.9.0

# Optional dependencies for enhanced functionality
seaborn>=0.11.0
```

## 📁 Structure du Projet

```
lstm-traffic-prediction/
├── main.py                        # 🚀 Launcher principal
├── tuning_lstm.py                  # 🔧 Optimisation d'hyperparamètres
├── lstm.py                         # 🧠 Entraînement et évaluation
├── analyse_similarite_lstm.py      # 📊 Analyse de similarité
├── requirements.txt                # 📦 Dépendances Python
├── README.md                       # 📖 Documentation
├── data/                           # 📂 Données d'entrée
│   ├── intersection1/
│   │   └── traffic_data.csv
│   ├── intersection2/
│   │   └── traffic_data.csv
│   └── ...
├── models/                         # 🤖 Modèles sauvegardés
│   ├── sensor_A12_Intersection1_bs64_hs128_nl2_do20_lr1e-3_ep50_ws24_mae125.pt
│   └── ...
└── results/                        # 📈 Résultats et visualisations
    ├── predictions/
    ├── comparisons/
    └── similarity_analysis/
```

## 🚀 Guide d'Utilisation

### Démarrage Rapide

#### 1. Interface Interactive (Recommandé)
```bash
python main.py
```

Cette commande lance l'interface interactive qui :
- Affiche le statut de tous les composants
- Vérifie les dépendances automatiquement
- Propose une sélection guidée des fonctionnalités

#### 2. Vérification du Système
```bash
python main.py --status
```

Affiche un diagnostic complet avec les dépendances manquantes et les instructions d'installation.

#### 3. Aide Contextuelle
```bash
python main.py --help-component tuning
python main.py --help-component lstm
python main.py --help-component similarity
```

### Workflow Typique

1. **Préparation des données** : Placer les fichiers CSV dans `data/`
2. **Optimisation** : `tuning_lstm.py` pour trouver les meilleurs hyperparamètres
3. **Entraînement** : `lstm.py` pour entraîner avec les paramètres optimaux
4. **Analyse** : `analyse_similarite_lstm.py` pour analyser les similarités

## 📘 Exemples Détaillés

### 🔧 1. Optimisation d'Hyperparamètres

#### Utilisation de Base
```bash
# Via le launcher
python main.py --component tuning --data ./data --trials 100

# Directement
python tuning_lstm.py --data ./data --trials 100
```

#### Configuration Avancée
```bash
# Optimisation intensive
python tuning_lstm.py \
    --data ./traffic_data \
    --trials 500 \
    --threshold 0.5
```

**Fonctionnalités :**
- **Optimisation multi-objectifs** : Minimise la MAE et la complexité du modèle
- **Validation croisée temporelle** : 5 folds avec TimeSeriesSplit
- **Parallélisation automatique** : Utilise tous les GPU disponibles
- **Early stopping** : Arrêt intelligent pour éviter le surapprentissage

**Sortie :**
```
models/
├── sensor_A12_Intersection1_bs64_hs128_nl2_do20_lr1e-3_ep50_ws24_mae125.pt
├── sensor_B23_Intersection2_bs32_hs96_nl3_do30_lr5e-4_ep40_ws12_mae110.pt
└── ...
```

### 🧠 2. Entraînement et Évaluation

#### Mode GUI (Interface Streamlit)
```bash
# Via le launcher
python main.py --component lstm

# Directement
python lstm.py
```

**Interface Web :** Ouvre automatiquement dans le navigateur avec :
- Upload de fichiers CSV par glisser-déposer
- Sélection interactive des capteurs
- Configuration des hyperparamètres via sliders
- Visualisations en temps réel
- Téléchargement des modèles entraînés

#### Mode CLI (Ligne de Commande)
```bash
# Entraînement en mode CLI
python main.py --component lstm --cli --data ./data --output ./results

# Mode verbeux
python lstm.py --cli --data ./data --output ./results --verbose

# Mode silencieux
python lstm.py --cli --data ./data --output ./results --quiet
```

**Workflow CLI :**
1. **Découverte automatique** des fichiers CSV
2. **Sélection interactive** des fichiers et capteurs
3. **Configuration guidée** des hyperparamètres
4. **Entraînement avec validation croisée** (optionnelle)
5. **Génération automatique** des visualisations
6. **Sauvegarde structurée** des résultats

#### Modes d'Utilisation Disponibles

##### **A. Entraîner Nouveau Modèle**
```bash
# Configuration interactive des hyperparamètres
Enter hidden_size [64]: 128
Enter num_layers [2]: 3
Enter dropout [0.2]: 0.3
Enter learning_rate [0.0005]: 0.001
Enter epochs [20]: 50
Enter window_size [12]: 24
```

##### **B. Charger Modèle Existant**
```bash
# Évaluation de modèles pré-entraînés
python lstm.py --cli
# Sélectionnez "Charger modèle existant"
# Choisissez les modèles .pt à évaluer
```

##### **C. Comparer Plusieurs Modèles**
```bash
# Comparaison de performances
python lstm.py --cli
# Sélectionnez "Comparer plusieurs modèles"
# Sélectionnez 2+ modèles pour le même capteur
```

### 📊 3. Analyse de Similarité

#### Interface Streamlit Dédiée
```bash
# Via le launcher
python main.py --component similarity

# Directement
streamlit run analyse_similarite_lstm.py
```

**Fonctionnalités de l'Interface :**

1. **Configuration Initiale**
   - Sélection des dossiers de données et modèles
   - Réglage du seuil de similarité (1-20% MAE)
   - Détection automatique du device (CPU/GPU)

2. **Analyse de Performance Croisée**
   - Matrice de performance interactive
   - Test de chaque modèle sur les données des autres capteurs
   - Calcul des différences relatives

3. **Visualisations Avancées**
   - **Heatmap interactive** : Matrice de performance avec échelle de couleurs
   - **Dendrogramme** : Clustering hiérarchique des modèles
   - **Tableaux récapitulatifs** : Performances détaillées par paire

4. **Recommandations Intelligentes**
   - Identification des modèles consolidables
   - Calcul du potentiel de réduction
   - Stratégies de groupement optimales

#### Exemple de Sortie d'Analyse

```
🎯 Modèles similaires trouvés (seuil: 5.0% MAE):

Groupe 1: Peut utiliser un modèle commun
- A12 (Intersection_Centre) - MAE original: 12.5%
- A15 (Intersection_Centre) - MAE original: 11.8%
- B03 (Intersection_Centre) - MAE original: 13.2%

Réduction potentielle: 2 modèles en moins (40% de réduction)
```

## 🏗 Architecture LSTM

### Modèle Neural Network

```python
class RegresseurLSTM(nn.Module):
    def __init__(self, in_size=6, hid_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hid_size, n_layers, 
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hid_size, 1)
```

### Features d'Entrée (6 dimensions)

1. **`flow`** : Flux du capteur (variable cible)
2. **`hour_cos`** : Encodage cyclique de l'heure (cos(2π×heure/24))
3. **`mean_flow_others`** : Flux moyen des autres capteurs (contexte spatial)
4. **`ma3`** : Moyenne mobile sur 3 périodes (tendance court terme)
5. **`ma6`** : Moyenne mobile sur 6 périodes (tendance moyen terme)
6. **`ma12`** : Moyenne mobile sur 12 périodes (tendance long terme)

### Hyperparamètres Optimisables

| Paramètre | Plage | Description |
|-----------|-------|-------------|
| `hidden_size` | 64-128 | Taille des couches cachées LSTM |
| `num_layers` | 2-4 | Nombre de couches LSTM empilées |
| `dropout` | 0.3-0.5 | Taux de dropout pour régularisation |
| `learning_rate` | 1e-4 à 1e-3 | Taux d'apprentissage Adam |
| `batch_size` | 16-128 | Taille des batchs d'entraînement |
| `window_size` | 12-24 | Longueur des séquences temporelles |
| `num_epochs` | 20-40 | Nombre d'époques d'entraînement |

## 📂 Datasets

### Format des Données d'Entrée

Les données doivent être organisées par intersection avec des fichiers CSV contenant :

```csv
count_point_name,measure_datetime,flow[veh/h]
A12,2025-07-01 00:00:00+02:00,145
A12,2025-07-01 01:00:00+02:00,89
A12,2025-07-01 02:00:00+02:00,67
A15,2025-07-01 00:00:00+02:00,203
A15,2025-07-01 01:00:00+02:00,156
...
```

### Structure Recommandée

```
data/
├── intersection_centre/
│   └── traffic_2025.csv
├── intersection_nord/
│   └── traffic_2025.csv
├── intersection_sud/
│   └── traffic_2025.csv
└── intersection_est/
    └── traffic_2025.csv
```

### Préprocessing Automatique

- **Nettoyage des timestamps** : Conversion UTC → Europe/Paris
- **Gestion des valeurs manquantes** : Imputation par moyenne
- **Validation des données** : Vérification de cohérence
- **Feature engineering** : Génération automatique des 6 features

## 📊 Métriques et Évaluation

### Métriques Principales

- **MAE (Mean Absolute Error)** : Erreur absolue moyenne
- **MAE%** : MAE normalisée par le flux moyen (interprétation intuitive)
- **MSE (Mean Squared Error)** : Erreur quadratique moyenne

### Validation

- **Validation croisée temporelle** : TimeSeriesSplit (5 folds)
- **Division temporelle** : 80% entraînement / 20% test
- **Early stopping** : Patience de 10 époques

## 🤝 Remerciements

Je tiens à remercier chaleureusement l'équipe du **Laboratoire CReSTIC - Lab-i*** de l'**Université de Reims Champagne-Ardennes** pour leur encadrement et leur soutien tout au long de ce stage de M1 :

- **Monsieur Fouchal** - Directeur de stage et encadrant principal
- **Monsieur Rabat** - Co-encadrant et expert en intelligence artificielle  
- **Monsieur Ninet** - Conseiller technique et spécialiste des réseaux
- **Monsieur Keziou** - Expert en analyse de données et statistiques

Leur expertise, leurs conseils avisés et leur disponibilité ont été essentiels à la réussite de ce projet de recherche appliquée.

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📞 Support et Contact

- **Auteur :** Jules Lefèvre
- **Email :** jules.lefevre@etudiant.univ-reims.fr
- **Institution :** Université de Reims Champagne-Ardennes
- **Laboratoire :** CReSTIC - Lab-i*

Pour toute question, suggestion ou contribution, n'hésitez pas à ouvrir une issue ou à me contacter directement.

---

**Fait avec ❤️ dans le cadre du M1 Intelligence Artificielle - URCA 2025**