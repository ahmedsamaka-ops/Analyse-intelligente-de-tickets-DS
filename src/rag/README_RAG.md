# Exemples d'utilisation du template RAG

## 1. Test rapide avec contexte bidon

Pour tester le template RAG avec du contexte bidon, exécutez :

```powershell
# Activez l'environnement virtuel
.venv\Scripts\Activate.ps1

# Testez le template RAG
python src/rag/test_rag_template.py
```

Ce script va :
- ✅ Tester une question dont la réponse EST dans le contexte
- ✅ Tester une question HORS contexte (devrait dire "je ne sais pas")
- ✅ Tester avec des expressions marocaines
- 📊 Générer un rapport de test dans `rag_template_test_results.txt`

## 2. Utilisation dans votre code

```python
from src.rag.chatbot import chat_with_context

# Votre contexte (récupéré depuis la base de données vectorielle)
contexte = """
Document: Problème réseau
Le problème de connexion a été résolu en redémarrant le routeur.
Temps de résolution: 30 minutes.
"""

# Question de l'utilisateur
question = "Combien de temps a pris la résolution ?"

# Obtenir la réponse basée uniquement sur le contexte
reponse = chat_with_context(question, contexte)
print(reponse)  # Devrait dire: "30 minutes"
```

## 3. Structure du template RAG

Le template utilisé est :

```
CONTEXTE: {context}

QUESTION: {question}

INSTRUCTIONS: Réponds en te basant UNIQUEMENT sur le contexte fourni. 
Si tu ne trouves pas la réponse, dis "Je ne trouve pas cette information dans les documents."
```

## 4. Tests automatiques

Le script `test_rag_template.py` teste 4 scénarios :
1. ✅ Information présente dans le contexte
2. ❌ Question complètement hors contexte
3. ✅ Contexte avec expressions marocaines
4. ❌ Information non mentionnée

Score attendu : 4/4 tests passés
