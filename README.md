# 📊 Business Client Prediction — Prédiction du Désabonnement Client (Customer Churn)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)

---

## 🎯 Présentation du Projet

Ce projet a pour objectif de **prédire le désabonnement des clients** (*Customer Churn*) d'une entreprise de télécommunications. En identifiant à l'avance les clients susceptibles de résilier leur abonnement, l'entreprise peut mettre en place des **stratégies de rétention proactives** et réduire significativement son taux de churn.

> **Problème métier :** Acquérir un nouveau client coûte 5 à 7 fois plus cher que de conserver un client existant. La prédiction du churn est donc un levier stratégique majeur.

---

## 📁 Structure du Projet

```text
BusinessClientPrediction/
│
├── 📓 notebook_final.ipynb                    # Notebook final soigné (démarche complète)
│
├── 📂 drafts/                                 # Notebooks de brouillon (expérimentations)
│   ├── 1EDA.ipynb                             # Analyse Exploratoire des Données
│   └── 2Preprocessing.ipynb                   # Prétraitement + ML + Deep Learning
│
├── 📊 WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset source (Telco Customer Churn - IBM)
├── 💾 scaler.pkl                              # StandardScaler sauvegardé (généré)
├── 🧠 ann_model.h5                            # Modèle ANN sauvegardé (généré)
└── 📄 README.md                               # Ce fichier
```

---

## 📊 Dataset

**Source :** [Telco Customer Churn — IBM Watson Analytics](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Caractéristique | Valeur |
|---|---|
| Nombre de clients | **7 043** |
| Nombre de variables | **21** |
| Variable cible | `Churn` (Yes/No) |
| Taux de churn | ~**26.5%** |

**Variables clés :**
- **Démographiques :** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services :** `PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV`...
- **Contrat & Facturation :** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Cible :** `Churn` — Si le client a résilié son abonnement

---

## 🗂️ Description des Notebooks

### 📓 `notebook_final.ipynb` — Le Notebook Principal

Ce notebook est le document final et propre du projet. Il regroupe l'intégralité de la démarche, divisée en 3 grandes parties :

#### 🔍 Partie 1 : Analyse Exploratoire (EDA)
- **Inspection** : Types de données, statistiques descriptives, valeurs manquantes.
- **Variable cible** : Distribution du churn, impact du déséquilibre des classes (~73% vs 27%).
- **Insights clés** : 
  1. Les clients avec contrat **mois à mois** churns à ~42%.
  2. Les **nouveaux clients** (< 12 mois) sont très vulnérables.
  3. Les charges mensuelles **élevées** augmentent le risque.

#### 🛠️ Partie 2 : Prétraitement des Données
- **Nettoyage** : Correction de `TotalCharges` et imputation par la médiane.
- **Feature Engineering** : Création de variables expertes (`AverageMonthlyCost`, `HighValueCustomer`, `LongTermCustomer`).
- **Préparation ML** : One-Hot Encoding, séparation Train/Test (80/20 stratifiée), et Standardisation.

#### 🤖 Partie 3 : Modélisation (ML & Deep Learning)
- **Machine Learning** : Entraînement de la **Régression Logistique** (baseline), **Random Forest** (avec calcul de la Feature Importance), et **XGBoost** (avec optimisation par GridSearchCV).
- **Deep Learning (ANN)** : Création d'un réseau de neurones avec TensorFlow/Keras (couches Dense, Dropout, Early Stopping).
- **Conclusion** : Comparaison des courbes ROC et sélection de XGBoost comme meilleur modèle (ROC-AUC ~0.84).

---

### 📂 Dossier `drafts/` — Les Brouillons d'Expérimentation

Ce dossier contient l'historique de recherche et les essais préliminaires :
- **`1EDA.ipynb`** : Les tous premiers tests de visualisation et d'exploration brute.
- **`2Preprocessing.ipynb`** : Les tests progressifs d'algorithmes, de nettoyage et d'optimisation avant la consolidation finale.

---

## 📈 Résultats Comparatifs

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Régression Logistique | ~0.80 | ~0.65 | ~0.55 | ~0.60 | ~0.83 |
| Random Forest | ~0.79 | ~0.65 | ~0.50 | ~0.57 | ~0.82 |
| **XGBoost** ⭐ | **~0.79** | **~0.65** | **~0.51** | **~0.57** | **~0.84** |
| ANN (Deep Learning) | ~0.79 | ~0.64 | ~0.53 | ~0.58 | ~0.83 |

### 🏆 Modèle Final : **XGBoost**

XGBoost a été sélectionné comme modèle final car il offre le meilleur compromis entre **ROC-AUC**, **F1-Score** et **interprétabilité**.

---

## 🚀 Installation et Utilisation

### Prérequis

- Python 3.9+
- Jupyter Notebook ou VS Code avec l'extension Jupyter

### Installation

```bash
# Créer et activer l'environnement virtuel 
python -m venv .venv
.venv\Scripts\activate   # Windows

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow joblib
```

### Lancement

```bash
# Ouvrir le notebook final (recommandé pour voir l'ensemble du projet) :
jupyter notebook notebook_final.ipynb

# (Optionnel) Pour consulter les expérimentations et brouillons :
jupyter notebook drafts/1EDA.ipynb
jupyter notebook drafts/2Preprocessing.ipynb
```

### Ordre de lecture recommandé

```
1️⃣  notebook_final.ipynb         → Démarche complète, résultats finaux et conclusion (à lire en priorité)
2️⃣  drafts/1EDA.ipynb            → (Optionnel) Détail des premières explorations de données
3️⃣  drafts/2Preprocessing.ipynb  → (Optionnel) Détail des tests d'algorithmes et d'optimisations
```

---

## 🛠️ Stack Technologique

| Outil | Version | Usage |
|---|---|---|
| Python | 3.9+ | Langage principal |
| Pandas | 2.x | Manipulation des données |
| NumPy | 1.x | Calculs numériques |
| Matplotlib / Seaborn | — | Visualisation |
| Scikit-learn | 1.x | ML, preprocessing, évaluation |
| XGBoost | 2.x | Gradient Boosting |
| TensorFlow / Keras | 2.x | Deep Learning (ANN) |
| Joblib | — | Sauvegarde du scaler |

---

## 📄 Licence

Ce projet est à usage académique et pédagogique.

---

*Projet réalisé dans le cadre d'une étude de cas en Data Science et Machine Learning.*
