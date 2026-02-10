# Configuration du Chatbot LLM

## Options disponibles

### Option 1: Ollama (Recommandé - Gratuit et Local) 🏠

**Installation:**
```bash
# Téléchargez Ollama depuis: https://ollama.ai/download
# Windows: téléchargez l'installeur

# Après installation, lancez Ollama
ollama serve

# Dans un autre terminal, téléchargez un modèle
ollama pull llama3.2
```

**Configuration:**
```bash
# Créez le fichier .env
cp .env.example .env

# Modifiez .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

### Option 2: OpenAI (Payant) 💰

**Obtenir la clé API:**
1. Allez sur https://platform.openai.com/api-keys
2. Créez une clé API
3. Copiez la clé

**Configuration:**
```bash
# Dans .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-votre-cle-ici
```

### Option 3: Mistral AI (Payant) 💰

**Obtenir la clé API:**
1. Allez sur https://console.mistral.ai/
2. Créez une clé API
3. Copiez la clé

**Configuration:**
```bash
# Dans .env:
LLM_PROVIDER=mistral
MISTRAL_API_KEY=votre-cle-ici
```

## Installation des dépendances

```bash
# Activez votre environnement virtuel
.venv\Scripts\Activate.ps1

# Installez les packages nécessaires
pip install python-dotenv requests

# Si vous utilisez OpenAI:
pip install openai

# Si vous utilisez Mistral:
pip install mistralai
```

## Test du chatbot

```bash
python src/rag/chatbot.py
```

Le test posera la question "Quelle est la capitale du Maroc ?" et enregistrera le résultat dans `llm_status.txt`.

## Utilisation dans votre code

```python
from src.rag.chatbot import chat

# Envoyez un message
response = chat("Comment résoudre un problème de connexion ?")
print(response)
```
