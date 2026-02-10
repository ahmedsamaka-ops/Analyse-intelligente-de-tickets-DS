"""
Script de démarrage rapide pour tester la fonction ask_bot
Initialise ChromaDB et lance un test
"""
import os
import sys

def main():
    """
    Script principal qui:
    1. Vérifie/installe les dépendances
    2. Initialise la base de connaissances
    3. Teste ask_bot
    """
    print("=" * 80)
    print(" DÉMARRAGE RAPIDE - Test ask_bot (RAG)")
    print("=" * 80)
    
    # Étape 1: Vérifier sentence-transformers
    print("\n📦 Vérification des dépendances...")
    try:
        import sentence_transformers
        print("   ✅ sentence-transformers installé")
    except ImportError:
        print("   ❌ sentence-transformers non trouvé")
        print("   📥 Installation en cours...")
        os.system('pip install sentence-transformers')
        import sentence_transformers
        print("   ✅ sentence-transformers installé")
    
    # Étape 2: Initialiser la base de données
    print("\n🔧 Initialisation de la base de connaissances...")
    from init_simple_db import init_simple_db
    init_simple_db()
    
    # Étape 3: Test de la question demandée
    print("\n" + "=" * 80)
    print(" TEST DE LA QUESTION")
    print("=" * 80)
    
    from chatbot import ask_bot
    
    question = "Comment résoudre un problème de connexion Maroc Telecom?"
    print(f"\n❓ Question: {question}\n")
    
    result = ask_bot(question)
    
    print(f"\n🤖 Réponse:")
    print("-" * 80)
    print(result['answer'])
    print("-" * 80)
    
    if result['sources']:
        print(f"\n📚 Sources utilisées:")
        for i, (source, distance) in enumerate(zip(result['sources'], result.get('distances', [])), 1):
            similarity = (1 - distance) * 100 if distance else 0
            print(f"   {i}. {source} (similarité: {similarity:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ Test terminé avec succès!")
    print("=" * 80)
    
    # Menu d'options
    print("\n📋 Options disponibles:")
    print("   1. Lancer les tests complets: python src/rag/test_ask_bot.py")
    print("   2. Mode interactif: python src/rag/test_ask_bot.py demo")
    print("   3. Utiliser dans votre code:")
    print("      from src.rag.chatbot import ask_bot")
    print("      result = ask_bot('votre question')")
    print("      print(result['answer'])")
    

if __name__ == "__main__":
    # Changer le répertoire de travail vers src/rag
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
