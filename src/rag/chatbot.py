"""
Chatbot pour le support technique - Contexte marocain
Supports: OpenAI, Mistral AI, ou Ollama (local)
"""
import os
from dotenv import load_dotenv
import json
import pickle
import numpy as np

# Charger les variables d'environnement
load_dotenv()

# Configuration
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "openai", "mistral", ou "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Configuration de la base de connaissances
DB_PATH = "data/simple_db"
EMBEDDINGS_FILE = os.path.join(DB_PATH, "embeddings.pkl")
DOCUMENTS_FILE = os.path.join(DB_PATH, "documents.json")

# Prompt système adapté au contexte marocain
SYSTEM_PROMPT = """Tu es un assistant support utile qui connaît le contexte marocain (français arabe dialecte).
Tu es spécialisé dans l'aide au support technique et tu comprends les expressions et le contexte culturel marocain.
Tu peux répondre en français et comprendre l'arabe dialecte marocain (darija) écrit en caractères latins.
Sois courtois, précis et efficace dans tes réponses."""

# Template RAG pour les questions avec contexte
RAG_TEMPLATE = """CONTEXTE: {context}

QUESTION: {question}

INSTRUCTIONS: Réponds en te basant UNIQUEMENT sur le contexte fourni. Si tu ne trouves pas la réponse, dis "Je ne trouve pas cette information dans les documents."
"""


def chat_with_openai(message: str) -> str:
    """Chat avec OpenAI GPT"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur OpenAI: {str(e)}"


def chat_with_mistral(message: str) -> str:
    """Chat avec Mistral AI"""
    try:
        from mistralai import Mistral
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur Mistral: {str(e)}"


def chat_with_ollama(message: str) -> str:
    """Chat avec Ollama (local)"""
    try:
        import requests
        
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"Erreur Ollama: {str(e)}\nAssurez-vous qu'Ollama est installé et en cours d'exécution (ollama serve)"


def chat(message: str) -> str:
    """
    Envoie un message au chatbot selon le provider configuré
    """
    if PROVIDER == "openai":
        return chat_with_openai(message)
    elif PROVIDER == "mistral":
        return chat_with_mistral(message)
    elif PROVIDER == "ollama":
        return chat_with_ollama(message)
    else:
        return f"Provider '{PROVIDER}' non reconnu. Options: openai, mistral, ollama"


def chat_with_context(question: str, context: str) -> str:
    """
    Envoie une question avec contexte au chatbot (RAG)
    
    Args:
        question: La question de l'utilisateur
        context: Le contexte/document pertinent
    
    Returns:
        La réponse basée uniquement sur le contexte
    """
    prompt = RAG_TEMPLATE.format(context=context, question=question)
    return chat(prompt)


def ask_bot(query: str, n_results: int = 3) -> dict:
    """
    Fonction RAG complète : recherche dans la base de connaissances + génération de réponse
    
    Args:
        query: La question de l'utilisateur
        n_results: Nombre de documents à récupérer (défaut: 3)
    
    Returns:
        dict avec 'answer', 'sources', et 'documents'
    """
    print(f"🔍 Recherche dans la base de connaissances...")
    
    try:
        # Vérifier que les fichiers existent
        if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(DOCUMENTS_FILE):
            return {
                "answer": "❌ La base de connaissances n'est pas initialisée. Exécutez d'abord: python src/rag/init_simple_db.py",
                "sources": [],
                "documents": []
            }
        
        # Étape A : Charger les données et chercher les documents similaires
        from sentence_transformers import SentenceTransformer
        
        # Charger le modèle
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Charger les embeddings et documents
        with open(EMBEDDINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            embeddings = data['embeddings']
        
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            all_documents = json.load(f)
        
        # Encoder la query
        query_embedding = model.encode([query])[0]
        
        # Calculer les similarités cosine
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Trier par similarité et prendre les top N
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        # Vérifier si des documents ont été trouvés
        if len(top_indices) == 0:
            return {
                "answer": "Je ne trouve aucun document pertinent dans ma base de connaissances.",
                "sources": [],
                "documents": []
            }
        
        # Récupérer les documents pertinents
        relevant_docs = [all_documents[idx] for idx in top_indices]
        documents = [doc["text"] for doc in relevant_docs]
        ids = [doc["id"] for doc in relevant_docs]
        distances = [1 - similarities[idx] for idx in top_indices]  # Convertir similarité en distance
        
        print(f"   ↪ {len(documents)} documents trouvés")
        for doc_id, distance in zip(ids, distances):
            similarity = (1 - distance) * 100
            print(f"      • {doc_id} (similarité: {similarity:.1f}%)")
        
        # Étape B : Construire le contexte avec les documents trouvés
        context_parts = []
        for i, (doc, doc_id) in enumerate(zip(documents, ids), 1):
            context_parts.append(f"[Document {i} - {doc_id}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        print(f"\n💬 Envoi de la question au LLM...")
        
        # Étape C : Envoyer au LLM avec le template RAG
        answer = chat_with_context(query, context)
        
        # Étape D : Retourner la réponse avec métadonnées
        return {
            "answer": answer,
            "sources": ids,
            "documents": documents,
            "distances": distances
        }
    
    except Exception as e:
        error_msg = str(e)
        return {
            "answer": f"❌ Erreur lors de la recherche: {error_msg}",
            "sources": [],
            "documents": []
        }


def test_chatbot():
    """
    Test simple du chatbot
    """
    print("=" * 60)
    print("TEST DU CHATBOT")
    print(f"Provider: {PROVIDER}")
    print("=" * 60)
    
    # Test 1: Question simple sur le Maroc
    question = "Quelle est la capitale du Maroc ?"
    print(f"\n❓ Question: {question}")
    print(f"🤖 Réponse: ", end="", flush=True)
    
    response = chat(question)
    print(response)
    
    # Test 2: Question technique
    question2 = "Comment je peux réinitialiser mon mot de passe ?"
    print(f"\n❓ Question: {question2}")
    print(f"🤖 Réponse: ", end="", flush=True)
    
    response2 = chat(question2)
    print(response2)
    
    # Test 3: Template RAG avec contexte bidon
    print("\n" + "=" * 60)
    print("TEST DU TEMPLATE RAG (avec contexte)")
    print("=" * 60)
    
    context_bidon = """
    Document support - Réinitialisation mot de passe
    
    Pour réinitialiser votre mot de passe:
    1. Allez sur la page de connexion
    2. Cliquez sur "Mot de passe oublié"
    3. Entrez votre email professionnel
    4. Vous recevrez un lien de réinitialisation dans les 5 minutes
    5. Le lien est valable pendant 24 heures
    
    Note: Si vous ne recevez pas l'email, vérifiez votre dossier spam.
    Pour toute assistance, contactez le support au +212-5XX-XXXXXX.
    """
    
    question_rag = "Combien de temps le lien de réinitialisation est-il valable ?"
    print(f"\n📄 Contexte: {context_bidon[:100]}...")
    print(f"\n❓ Question: {question_rag}")
    print(f"🤖 Réponse: ", end="", flush=True)
    
    response_rag = chat_with_context(question_rag, context_bidon)
    print(response_rag)
    
    # Test 4: Question hors contexte pour vérifier que le LLM répond bien qu'il ne sait pas
    question_hors_contexte = "Quelle est la capitale de la France ?"
    print(f"\n❓ Question (hors contexte): {question_hors_contexte}")
    print(f"🤖 Réponse: ", end="", flush=True)
    
    response_hors = chat_with_context(question_hors_contexte, context_bidon)
    print(response_hors)
    
    print("\n" + "=" * 60)
    print("✅ LLM fonctionne correctement !")
    print("=" * 60)
    
    # Enregistrer le statut
    with open("llm_status.txt", "w", encoding="utf-8") as f:
        f.write(f"LLM fonctionne\n")
        f.write(f"Provider: {PROVIDER}\n")
        f.write(f"Date de test: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n=== Test basique ===\n")
        f.write(f"Question: {question}\n")
        f.write(f"Réponse: {response}\n")
        f.write(f"\n=== Test RAG avec contexte ===\n")
        f.write(f"Question: {question_rag}\n")
        f.write(f"Réponse: {response_rag}\n")
        f.write(f"\n=== Test question hors contexte ===\n")
        f.write(f"Question: {question_hors_contexte}\n")
        f.write(f"Réponse: {response_hors}\n")
    
    print("\n📝 Statut enregistré dans llm_status.txt")


if __name__ == "__main__":
    test_chatbot()
