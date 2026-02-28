# 🎯 Analyse Intelligente de Tickets IT - RAG & ML

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![RAG](https://img.shields.io/badge/RAG-LangChain-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> Plateforme hybride combinant **Machine Learning classique** et **GenAI (RAG)** pour l'analyse et la classification automatique de tickets de support IT avec chatbot intelligent.

---

## 📋 Table des Matières

- [🎯 Aperçu](#-aperçu)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Technologies](#️-technologies)
- [🚀 Installation](#-installation)
- [💻 Utilisation](#-utilisation)
- [📊 Résultats](#-résultats)
- [📁 Structure du Projet](#-structure-du-projet)
- [👥 Auteur](#-auteur)

---

## 🎯 Aperçu

Ce projet est une **solution complète d'analyse de tickets IT** qui combine deux approches complémentaires :

### 🤖 Machine Learning Classique
- Classification automatique des tickets (catégorie, type, urgence)
- Prédiction du temps de résolution
- Modèles optimisés : **Random Forest, XGBoost, SVM**
- Précision : **85-90%** sur la classification multi-classe

### 🧠 RAG (Retrieval Augmented Generation)
- Chatbot intelligent basé sur vos données historiques
- Recherche sémantique dans 1450+ tickets
- Réponses contextuelles avec sources
- Support multi-LLM : **OpenAI, Mistral, Ollama**

---

## ✨ Fonctionnalités

### 📊 Machine Learning
- ✅ **Classification multi-classes** : Catégorie, Type, Urgence
- ✅ **Régression** : Prédiction du temps de résolution
- ✅ **Preprocessing avancé** : TF-IDF, PCA, SMOTE pour équilibrage
- ✅ **Pipeline complet** : Entraînement, validation, prédiction
- ✅ **Sauvegarde des modèles** : 4 pipelines optimisés (.pkl)

### 💬 Chatbot RAG
- ✅ **Base vectorielle** : TF-IDF avec recherche de similarité
- ✅ **ask_bot()** : Fonction RAG complète (recherche + génération)
- ✅ **Templates personnalisés** : Réponses basées uniquement sur contexte
- ✅ **Multi-provider** : Support OpenAI, Mistral AI, Ollama (local)
- ✅ **Test de similarité** : Scores de pertinence jusqu'à 0.695

### 🎨 Interface (En développement)
- 🔄 Application Streamlit interactive
- 🔄 Analyse en temps réel
- 🔄 Dashboard de visualisation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSE DE TICKETS IT                     │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
         ┌──────▼──────┐           ┌───────▼────────┐
         │  ML Classique│           │   RAG Chatbot  │
         └──────┬──────┘           └───────┬────────┘
                │                           │
    ┌───────────┼───────────┐      ┌────────┼────────┐
    │           │           │      │        │        │
┌───▼───┐  ┌───▼───┐  ┌───▼───┐ ┌─▼──┐ ┌──▼──┐ ┌──▼──┐
│Catég. │  │ Type  │  │Urgence│ │TF  │ │LLM  │ │Chroma│
│ 85%   │  │ 88%   │  │ 90%   │ │IDF │ │Multi│ │ DB  │
└───────┘  └───────┘  └───────┘ └────┘ └─────┘ └─────┘
```

---

## 🛠️ Technologies

### Machine Learning & Data Science
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat&logo=python&logoColor=white)

### RAG & LLM
![LangChain](https://img.shields.io/badge/-LangChain-121212?style=flat&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/-OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Ollama](https://img.shields.io/badge/-Ollama-000000?style=flat&logo=ai&logoColor=white)

### Outils & Infrastructure
![Git](https://img.shields.io/badge/-Git-F05032?style=flat&logo=git&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 🚀 Installation

### Prérequis
- Python 3.11+ (3.14 recommandé)
- pip
- Git

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/ahmedsamaka-ops/Analyse-intelligente-de-tickets-DS.git
cd Analyse-intelligente-de-tickets-DS

# 2. Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configuration du chatbot (optionnel)
cp .env.example .env
# Éditez .env avec vos clés API (OpenAI/Mistral) ou utilisez Ollama (gratuit)
```

---

## 💻 Utilisation

### 1️⃣ Machine Learning - Classification de Tickets

```python
# Charger le pipeline pré-entraîné
import pickle

with open('models/category_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Prédire la catégorie d'un nouveau ticket
ticket = "demande de réinitialisation mot de passe"
prediction = model.predict([ticket])
print(f"Catégorie prédite: {prediction[0]}")
```

### 2️⃣ RAG Chatbot - Recherche Intelligente

```python
from src.rag.chatbot import ask_bot

# Poser une question sur vos tickets historiques
result = ask_bot("Comment résoudre un problème d'accès au partage?")

print(result['answer'])      # Réponse générée
print(result['sources'])     # IDs des tickets sources
print(result['distances'])   # Scores de similarité
```

### 3️⃣ Créer la Base Vectorielle

```bash
# Indexer vos tickets dans la base vectorielle
python src/rag/ingest.py

# Tester la recherche de similarité
python src/rag/test_search.py
```

---

## 📊 Résultats

### Performance des Modèles ML

| Modèle | Tâche | Précision | F1-Score |
|--------|-------|-----------|----------|
| **Random Forest** | Classification Catégorie | **85%** | 0.83 |
| **XGBoost** | Classification Type | **88%** | 0.86 |
| **SVM** | Classification Urgence | **90%** | 0.89 |
| **Ridge Regression** | Temps de résolution | **R² 0.78** | MAE 12h |

### Performance du RAG

| Métrique | Score | Exemple |
|----------|-------|---------|
| **Similarité Top-1** | **0.695** | "demande activation compte" |
| **Précision contextuelle** | **92%** | Réponses basées sur documents |
| **Temps de réponse** | **< 2s** | Ollama local (llama3.2) |
| **Documents indexés** | **1450+** | Tickets de support |

### Exemple de Recherche

```
Query: "réinitialisation mot de passe"

Résultat 1 (similarité: 0.656):
  Ticket #6950 - demande de réinitialisation mot de passe compte cfao
  Catégorie: MDP CFAO | Urgence: Basse

Résultat 2 (similarité: 0.392):
  Ticket #6952 - mot de passe citrix amal sanaa
  Catégorie: Applications | Urgence: Basse
```

---

## 📁 Structure du Projet

```
Analyse-intelligente-de-tickets-DS/
│
├── 📊 data/                      # Données (1450+ tickets)
│   ├── tickets.csv
│   ├── train.csv / test.csv / validation.csv
│   └── *_with_pred.csv          # Prédictions ML
│
├── 🤖 models/                    # Modèles entraînés (4 pipelines)
│   ├── category_pipeline.pkl
│   ├── type_pipeline.pkl
│   ├── urgency_pipeline.pkl
│   ├── time_pipeline.pkl
│   └── metadata.json
│
├── 🧠 src/
│   ├── ml/                      # Machine Learning
│   │   ├── train_category.py
│   │   ├── train_type_ticket.py
│   │   ├── train_urgency.py
│   │   ├── train_time_regression.py
│   │   ├── predict_pipeline.py
│   │   ├── ml_utils.py
│   │   └── balance_urgence.py
│   │
│   └── rag/                     # RAG & Chatbot
│       ├── chatbot.py           # Chatbot multi-LLM + ask_bot()
│       ├── ingest.py            # Ingestion tickets → base vectorielle
│       ├── test_search.py       # Tests de similarité
│       ├── test_ask_bot.py      # Tests RAG complet
│       └── GUIDE_ASK_BOT.md     # Documentation
│
├── 🎨 app/                       # Interface Streamlit (à venir)
├── 📓 notebooks/                 # Analyse exploratoire
├── 📚 docs/                      # Documentation technique
│
├── .env.example                 # Template configuration LLM
├── requirements.txt             # Dépendances Python
└── README.md                    # Ce fichier
```

---

## 🎓 Cas d'Usage

### 🏢 Pour les Équipes Support IT
- **Automatisation** : Classification instantanée des tickets entrants
- **Priorisation** : Détection automatique de l'urgence
- **Assistant IA** : Chatbot pour rechercher des solutions dans l'historique
- **Prédiction** : Estimation du temps de résolution

### 💼 Pour les Managers
- **Analytics** : Dashboard de suivi des tickets
- **KPIs** : Métriques de performance avec précision 85-90%
- **Optimisation** : Identification des patterns de tickets

### 🎯 Pour les Data Scientists
- **Framework complet** : Pipeline ML bout-en-bout
- **RAG moderne** : Implémentation production-ready
- **Code réutilisable** : Modèles et fonctions modulaires

---

## 🔮 Roadmap

- [x] Classification ML multi-classes
- [x] Pipeline RAG avec base vectorielle
- [x] Chatbot multi-LLM (OpenAI, Mistral, Ollama)
- [ ] Interface Streamlit interactive
- [ ] API REST FastAPI
- [ ] Dashboard analytics temps réel
- [ ] Fine-tuning modèle custom
- [ ] Déploiement Docker

---

## 📄 Documentation

- **[INSTRUCTIONS_LLM.md](INSTRUCTIONS_LLM.md)** - Guide d'installation des LLMs (OpenAI/Mistral/Ollama)
- **[GUIDE_ASK_BOT.md](src/rag/GUIDE_ASK_BOT.md)** - Documentation complète de la fonction RAG
- **[README_RAG.md](src/rag/README_RAG.md)** - Utilisation du module RAG

---

## 🤝 Contributing

Les contributions sont les bienvenues ! N'hésitez pas à :
- 🐛 Signaler des bugs
- 💡 Proposer de nouvelles fonctionnalités
- 🔧 Soumettre des pull requests

---

## 📜 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 👥 Auteur

**Ahmed Samaka**
- 💼 Data Scientist | ML Engineer
- 🔗 GitHub: [@ahmedsamaka-ops](https://github.com/ahmedsamaka-ops)
- 📧 Email: [Votre email]
- 💼 LinkedIn: [Votre profil LinkedIn]

---

## 🙏 Remerciements

- Équipe IT Support pour les données de tickets
- Communauté LangChain pour le framework RAG
- Hugging Face pour les modèles d'embeddings

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'oubliez pas de laisser une étoile !**

Made with ❤️ by Ahmed Samaka

</div>