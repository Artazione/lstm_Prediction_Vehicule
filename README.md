# 🚦 Prédiction de Flux Véhicules avec Modèles LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-orange.svg)](https://streamlit.io/)

> **Projet réalisé dans le cadre de mon stage de M1 Informatique Parcours Intelligence Artificielle**  
> **Laboratoire Lab-i*** - Université de Reims Champagne-Ardennes  
> **Période :** Avril --> Juillet 2025

Ce projet développé dans le cadre d'un stage de M1 Informatique Parcours Intelligence Artificielle à l'Université de Reims Champagne-Ardenne (Lab-i*) propose une solution complète pour la prédiction de flux de véhicules utilisant des modèles LSTM. Il inclut l'optimisation d'hyperparamètres, l'entraînement de modèles et l'analyse de similarité entre capteurs.

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
  - [Launcher principal](#launcher-principal-mainpy)
  - [Optimisation d'hyperparamètres](#optimisation-dhyperparamètres-tuning_lstmpy)
  - [Entraînement et évaluation](#entraînement-et-évaluation-lstmpy)
  - [Analyse de similarité](#analyse-de-similarité-analyse_similarite_lstmpy)
- [Format des données](#format-des-données)
- [Exemples d'utilisation](#exemples-dutilisation)
- [Visualisations](#visualisations)
- [Remerciements](#remerciements)

## 🎯 Aperçu

Le projet propose une pipeline complète pour :

- **Optimisation automatique** d'hyperparamètres avec Optuna
- **Entraînement de modèles LSTM** spécialisés par capteur
- **Interface graphique** interactive (Streamlit) et **mode CLI**
- **Analyse de similarité** entre modèles pour identifier les possibilités de consolidation
- **Visualisations avancées** des performances et prédictions

### Fonctionnalités principales

- 🧠 **Modèles LSTM** avec feature engineering automatique (moyennes mobiles, encodage cyclique)
- 🔧 **Optimisation bayésienne** des hyperparamètres avec validation croisée temporelle
- 📊 **Interface web** intuitive avec Streamlit
- 💻 **Mode CLI** pour l'automatisation et la production
- 🔍 **Analyse de similarité** pour optimiser le nombre de modèles
- 📈 **Visualisations interactives** des performances et prédictions

## 🚀 Installation

### Prérequis

- Python 3.8+
- CUDA (optionnel, pour accélération GPU)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/Artazione/lstm_Prediction_Vehicule
cd lstm_Prediction_Vehicule

# Installer les dépendances
pip install -r requirements.txt
```

### Structure des données

Organisez vos données CSV (provenant de https://avatar.cerema.fr/) selon cette structure :

```
data/
├── Intersection1/
│   └── donnees_capteurs.csv
├── Intersection2/
│   └── donnees_capteurs.csv
└── ...
```

## 🏗️ Structure du projet

```
lstm_Prediction_Vehicule/
├── main.py                      # 🎮 Launcher principal
├── tuning_lstm.py               # 🔧 Optimisation d'hyperparamètres
├── lstm.py                      # 🧠 Entraînement et évaluation des modèles
├── analyse_similarite_lstm.py   # 🔍 Analyse de similarité entre modèles
├── requirements.txt             # 📦 Dépendances Python
├── data/                        # 📁 Dossier des données CSV
├── models/                      # 💾 Modèles entraînés (.pt)
└── resultats/                   # 📊 Résultats et visualisations
```

## 📖 Utilisation

### Launcher principal (`main.py`)

Le point d'entrée unique qui permet d'accéder à tous les composants du projet.

#### Interface interactive
```bash
python main.py
```

#### Lancement direct d'un composant
```bash
# Optimisation d'hyperparamètres
python main.py --component tuning --data ./data --trials 50

# Entraînement de modèles (interface web)
python main.py --component lstm --data ./data

# Entraînement de modèles (mode CLI)
python main.py --component lstm --cli --data ./data --output ./resultats

# Analyse de similarité (interface web)
python main.py --component similarity

# Analyse de similarité (mode CLI)
python main.py --component similarity --cli --data ./data --models ./models --output ./resultats
```

#### Aide et statut
```bash
# Vérifier le statut des composants
python main.py --status

# Aide détaillée pour un composant
python main.py --help-component lstm
```

### Optimisation d'hyperparamètres (`tuning_lstm.py`)

Optimise automatiquement les hyperparamètres LSTM pour chaque capteur avec Optuna.

#### Utilisation
```bash
python tuning_lstm.py --data ./data --trials 100 --threshold 1.0
```

#### Paramètres

- `--data` : Dossier racine contenant les données d'intersections
- `--trials` : Nombre d'essais Optuna par capteur (défaut: 100)
- `--threshold` : Seuil MAE% pour privilégier la simplicité (défaut: 1.0)

#### Fonctionnalités

- **Optimisation bayésienne** avec Optuna
- **Validation croisée temporelle** (TimeSeriesSplit)
- **Parallélisation multi-GPU** automatique
- **Early stopping** pour éviter le surapprentissage
- **Sauvegarde automatique** avec nomenclature descriptive

#### Exemple de sortie
```
=== Intersection1 / CapteurA ===
Final: MAE%=2.34, params=1247
Modèle enregistré: models/sensor_CapteurA_Intersection1_bs64_hs96_nl2_do30_lr5e-04_ep25_ws24_mae234.pt
```

### Entraînement et évaluation (`lstm.py`)

Application complète pour l'entraînement et l'évaluation des modèles LSTM.

#### Mode GUI (Interface web)
```bash
python lstm.py
```

Fonctionnalités de l'interface web :
- 📁 **Upload de fichiers CSV** par glisser-déposer
- 🎛️ **Configuration interactive** des hyperparamètres
- 📊 **Visualisations en temps réel** des performances
- 💾 **Téléchargement automatique** des modèles
- 🔄 **Validation croisée optionnelle**

#### Mode CLI
```bash
python lstm.py --cli --data ./data --output ./resultats --verbose
```

Paramètres CLI :
- `--data` : Dossier des données CSV (défaut: ./data)
- `--output` : Dossier de sortie (défaut: ./resultats)
- `--verbose` : Mode verbeux
- `--quiet` : Mode silencieux

#### Modes d'utilisation CLI

1. **Entraîner nouveau modèle**
   - Configuration interactive des hyperparamètres
   - Sélection des capteurs à traiter
   - Validation croisée optionnelle

2. **Charger modèle existant**
   - Évaluation de modèles pré-entraînés
   - Génération de nouvelles prédictions
   - Comparaison des performances

3. **Comparer plusieurs modèles**
   - Analyse comparative de performance
   - Visualisations côte à côte
   - Identification du meilleur modèle

### Analyse de similarité (`analyse_similarite_lstm.py`)

Analyse la similarité entre modèles pour identifier les possibilités de consolidation.

#### Mode GUI (Interface web)
```bash
python analyse_similarite_lstm.py
```

Fonctionnalités de l'interface :
- 🔍 **Sélection interactive** des modèles à comparer
- 📊 **Matrice de performance croisée** interactive
- 🌳 **Dendrogramme de clustering** hiérarchique
- 💡 **Recommandations de consolidation** automatiques

#### Mode CLI
```bash
python analyse_similarite_lstm.py --cli --data ./data --models ./models --threshold 5.0 --output ./resultats
```

Paramètres CLI :
- `--data` : Dossier des données CSV
- `--models` : Dossier des modèles (.pt)
- `--threshold` : Seuil de similarité MAE% (défaut: 5.0)
- `--output` : Dossier de sortie des résultats

#### Fonctionnalités d'analyse

- **Performance croisée** : Test de chaque modèle sur les données des autres capteurs
- **Clustering hiérarchique** : Regroupement des modèles similaires
- **Métriques de consolidation** : Calcul du potentiel de réduction
- **Visualisations avancées** : Heatmaps, dendrogrammes, graphiques temporels

#### Exemple de sortie CLI
```
📊 ANALYSE DE REGROUPEMENT (Seuil: 5.0% MAE)
✅ 3 paire(s) de modèles similaires trouvée(s):

   1. CapteurA (Intersection1) ↔ CapteurB (Intersection1)
      Différence moyenne: 2.45% MAE

💡 RECOMMANDATIONS DE CONSOLIDATION:
   Groupe 1 - Peut utiliser un modèle commun:
     - CapteurA (Intersection1) - MAE original: 2.1%
     - CapteurB (Intersection1) - MAE original: 2.3%

📈 IMPACT DE LA CONSOLIDATION:
   • Réduction possible: 3 modèles en moins
   • Pourcentage de réduction: 25.0%
```

## 📁 Format des données

Les données CSV doivent provenir du site https://avatar.cerema.fr/ et contenir les colonnes suivantes :

```csv
count_point_name;measure_datetime;flow[veh/h]
CapteurA;2024-01-01T00:00:00+01:00;45
CapteurB;2024-01-01T00:00:00+01:00;67
...
```

### Colonnes requises

- `count_point_name` : Nom/ID du capteur
- `measure_datetime` : Timestamp au format ISO 8601
- `flow[veh/h]` : Flux de véhicules par heure

### Feature engineering automatique

Le système génère automatiquement les caractéristiques suivantes :

- **hour_cos** : Encodage cyclique de l'heure (cosinus)
- **mean_flow_others** : Flux moyen des autres capteurs
- **ma3, ma6, ma12** : Moyennes mobiles sur 3, 6 et 12 périodes

## 💡 Exemples d'utilisation

### Workflow complet

1. **Collecte des données**
   ```bash
   # Organiser les données par intersection
   mkdir -p data/Intersection1 data/Intersection2
   # Copier les fichiers CSV dans les dossiers appropriés
   ```

2. **Optimisation des hyperparamètres**
   ```bash
   python main.py --component tuning --data ./data --trials 50
   ```

3. **Entraînement des modèles**
   ```bash
   python main.py --component lstm --cli --data ./data --output ./resultats
   ```

4. **Analyse de similarité**
   ```bash
   python main.py --component similarity --cli --data ./data --models ./models --output ./resultats
   ```

### Cas d'usage spécifiques

#### Développement et expérimentation
```bash
# Interface graphique pour l'exploration interactive
python lstm.py
```

#### Production et automatisation
```bash
# Pipeline CLI complète
python tuning_lstm.py --data ./data --trials 100
python lstm.py --cli --data ./data --output ./models --quiet
python analyse_similarite_lstm.py --cli --data ./data --models ./models --output ./resultats
```

#### Analyse comparative
```bash
# Comparaison de plusieurs jeux d'hyperparamètres
python lstm.py --cli --data ./data --output ./resultats
# Sélectionner "Comparer plusieurs modèles"
```

## 📊 Visualisations

Le projet génère automatiquement plusieurs types de visualisations :

### Mode GUI (Streamlit)
- **Courbes d'entraînement** interactives
- **Prédictions vs réalité** avec sélection de date
- **Matrices de confusion** pour l'analyse d'erreurs
- **Dendrogrammes** de clustering
- **Heatmaps** de performance croisée

### Mode CLI
- **Graphiques PNG** haute résolution sauvegardés
- **Tableaux ASCII** formatés dans le terminal
- **Rapports CSV** détaillés
- **Logs complets** de l'entraînement

### Exemples de fichiers générés
```
resultats/
├── 20250124_143052/
│   ├── loss_capteur_A12_20250124_143052.png
│   ├── predictions_capteur_A12_2024-01-15_20250124_143052.png
│   ├── heatmap_performance_20250124_143052.png
│   ├── dendrogramme_20250124_143052.png
│   ├── resultats_croises_20250124_143052.csv
│   └── details_modeles_20250124_143052.csv
```

## 🔧 Configuration avancée

### Variables d'environnement
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Sélection des GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Gestion mémoire GPU
```

### Optimisation GPU
```python
# Dans tuning_lstm.py, ajuster selon votre matériel
MAX_WORKERS = 2  # Nombre de workers DataLoader
PATIENCE = 10    # Patience pour early stopping
```

### Personnalisation des hyperparamètres
Modifier les plages dans `tuning_lstm.py` :
```python
params = {
    'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
    'num_layers': trial.suggest_int('num_layers', 1, 5),
    # ...
}
```

## 🎯 Remerciements

Ce projet a été réalisé dans le cadre d'un stage de M1 Informatique Parcours Intelligence Artificielle à l'Université de Reims Champagne-Ardenne, au sein du laboratoire Lab-i*.

**Remerciements particuliers à :**
- **Monsieur Fouchal**
- **Monsieur Rabat** 
- **Monsieur Ninet**
- **Monsieur Keziou**

Pour leur encadrement, leurs conseils et leur soutien tout au long de ce projet.

---

### 📞 Support

Pour toute question ou problème :
- Consultez les logs d'erreur avec `--verbose`
- Vérifiez le statut des composants avec `python main.py --status`
- Assurez-vous que les données respectent le format requis
