# Guide d'utilisation de la fonction ask_bot (RAG Complet)

## 🎯 Vue d'ensemble

La fonction `ask_bot()` est une implémentation complète de RAG (Retrieval Augmented Generation) qui :
1. **Recherche** les documents pertinents dans ChromaDB
2. **Construit** un prompt avec le contexte trouvé
3. **Envoie** la requête au LLM (Ollama/OpenAI/Mistral)
4. **Retourne** une réponse basée uniquement sur les documents

## 🚀 Démarrage rapide (Test en 1 commande)

```powershell
# Activez l'environnement virtuel
.venv\Scripts\Activate.ps1

# Lancez le test rapide
python src/rag/quick_start.py
```

Ce script va :
- ✅ Vérifier les dépendances
- ✅ Initialiser ChromaDB avec 6 tickets de support
- ✅ Tester : "Comment résoudre un problème de connexion Maroc Telecom?"

## 📋 Étapes détaillées

### 1. Initialiser la base de connaissances

```powershell
python src/rag/init_chroma.py
```

Cela crée une base ChromaDB dans `data/chroma_db/` avec 6 tickets :
- Connexion Maroc Telecom
- Réinitialisation mot de passe
- VPN Orange
- Imprimante réseau
- Lenteur Inwi
- Email Outlook

### 2. Configurer le LLM

**Option A : Ollama (Gratuit, Local)**
```powershell
# Installez Ollama si ce n'est pas fait
# https://ollama.ai/download

ollama serve
ollama pull llama3.2
```

**Option B : OpenAI/Mistral**
```env
# Modifiez .env
LLM_PROVIDER=openai
OPENAI_API_KEY=votre-clé-ici
```

### 3. Tester ask_bot

**Test automatique avec 5 questions :**
```powershell
python src/rag/test_ask_bot.py
```

**Mode interactif (pose tes questions) :**
```powershell
python src/rag/test_ask_bot.py demo
```

## 💻 Utilisation dans votre code

```python
from src.rag.chatbot import ask_bot

# Poser une question
result = ask_bot("Comment résoudre un problème de connexion Maroc Telecom?")

# Afficher la réponse
print(result['answer'])

# Voir les sources utilisées
print(f"Sources: {result['sources']}")

# Accéder aux documents bruts
for doc in result['documents']:
    print(doc)
```

### Structure du résultat

```python
{
    "answer": "La réponse générée par le LLM",
    "sources": ["doc1", "doc2"],  # IDs des documents
    "documents": ["texte doc 1", "texte doc 2"],  # Textes complets
    "distances": [0.15, 0.23]  # Scores de similarité
}
```

## 🔧 Architecture de ask_bot

```
┌─────────────────┐
│  ask_bot(query) │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Étape A: Recherche ChromaDB        │
│ - Vectorisation de la query        │
│ - Recherche de similarité          │
│ - Récupération top N documents     │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Étape B: Construction du contexte  │
│ - Agrégation des documents         │
│ - Formatage avec le template RAG   │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Étape C: Envoi au LLM              │
│ - Application du template          │
│ - Appel API (OpenAI/Mistral/Ollama)│
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ Étape D: Retour du résultat        │
│ - Réponse + sources + métadonnées  │
└────────────────────────────────────┘
```

## 📊 Paramètres de ask_bot

```python
ask_bot(
    query: str,        # La question de l'utilisateur
    n_results: int = 3 # Nombre de documents à récupérer (défaut: 3)
) -> dict
```

## ❓ Questions de test suggérées

1. "Comment résoudre un problème de connexion Maroc Telecom?"
2. "Réinitialiser mon mot de passe oublié"
3. "Mon VPN ne fonctionne pas, que faire?"
4. "L'imprimante ne marche plus"
5. "Internet Inwi est très lent"
6. "Comment configurer Outlook?"

## 🔍 Résultats attendus

Avec la question **"Comment résoudre un problème de connexion Maroc Telecom?"** :

**Sources trouvées :** `doc1` (Ticket connexion Maroc Telecom)

**Réponse attendue :**
```
Pour résoudre un problème de connexion Maroc Telecom:
1. Vérifiez que le modem est bien allumé (voyant vert)
2. Redémarrez le modem (débranchez 30 secondes puis rebranchez)
3. Vérifiez les câbles RJ45 et RJ11
4. Si le problème persiste, appelez le 888 (service client Maroc Telecom)

Le temps de résolution moyen est de 2 heures.
```

## 🐛 Dépannage

**Erreur : "Collection does not exist"**
```powershell
# Initialisez la base d'abord
python src/rag/init_chroma.py
```

**Erreur : "Ollama connection failed"**
```powershell
# Lancez Ollama
ollama serve
```

**Pas de réponse pertinente**
- Vérifiez que les documents sont bien dans ChromaDB
- Augmentez `n_results` pour chercher plus de documents
- Ajoutez plus de tickets de support dans `init_chroma.py`

## 📝 Fichiers créés

- `src/rag/chatbot.py` - Fonctions LLM + ask_bot
- `src/rag/init_chroma.py` - Initialisation ChromaDB
- `src/rag/test_ask_bot.py` - Tests automatiques
- `src/rag/quick_start.py` - Script de démarrage rapide
- `data/chroma_db/` - Base de données vectorielle

## 🎓 Prochaines étapes

1. ✅ Testez ask_bot avec vos propres questions
2. 📚 Ajoutez vos propres tickets dans `init_chroma.py`
3. 🔧 Intégrez ask_bot dans votre application Streamlit
4. 🚀 Déployez avec vos vraies données de tickets
