"""
Test de la fonction ask_bot (RAG complet)
"""
from chatbot import ask_bot, PROVIDER

def test_ask_bot():
    """
    Test de la fonction RAG complète avec différentes questions
    """
    print("=" * 80)
    print(" TEST DE LA FONCTION ask_bot (RAG COMPLET)")
    print(f" Provider: {PROVIDER}")
    print("=" * 80)
    
    # Liste de questions de test
    test_questions = [
        "Comment résoudre un problème de connexion Maroc Telecom?",
        "Comment réinitialiser mon mot de passe?",
        "Mon VPN ne fonctionne pas, que faire?",
        "L'imprimante ne marche plus, comment faire?",
        "Internet Inwi est très lent, quelle solution?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*80}")
        print(f"\n❓ Question: {question}\n")
        
        # Appel de la fonction ask_bot
        result = ask_bot(question)
        
        # Affichage de la réponse
        print(f"\n🤖 Réponse:")
        print("-" * 80)
        print(result['answer'])
        print("-" * 80)
        
        # Affichage des sources
        if result['sources']:
            print(f"\n📚 Sources utilisées:")
            for j, (source, distance) in enumerate(zip(result['sources'], result.get('distances', [])), 1):
                similarity = (1 - distance) * 100 if distance else 0
                print(f"   {j}. {source} (similarité: {similarity:.1f}%)")
        else:
            print(f"\n📚 Aucune source trouvée")
        
        print("\n" + "="*80)
        
        # Petit délai entre les questions pour la lisibilité
        if i < len(test_questions):
            input("\n⏸️  Appuyez sur Entrée pour la question suivante...")
    
    print("\n✅ Tests terminés!")
    
    # Sauvegarder les résultats
    with open("ask_bot_test_results.txt", "w", encoding="utf-8") as f:
        f.write("Tests de la fonction ask_bot (RAG complet)\n")
        f.write(f"Provider: {PROVIDER}\n")
        f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, question in enumerate(test_questions, 1):
            f.write(f"\n{'='*60}\n")
            f.write(f"TEST {i}: {question}\n")
            f.write(f"{'='*60}\n\n")
            
            result = ask_bot(question)
            
            f.write(f"Réponse:\n{result['answer']}\n\n")
            
            if result['sources']:
                f.write(f"Sources: {', '.join(result['sources'])}\n")
            else:
                f.write("Sources: Aucune\n")
    
    print("\n📝 Résultats sauvegardés dans: ask_bot_test_results.txt")


def demo_ask_bot():
    """
    Démonstration interactive de ask_bot
    """
    print("=" * 80)
    print(" DÉMONSTRATION INTERACTIVE ask_bot")
    print("=" * 80)
    print("\nTapez vos questions (ou 'quit' pour quitter)")
    print("Exemple: Comment résoudre un problème de connexion?\n")
    
    while True:
        question = input("❓ Votre question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Au revoir!")
            break
        
        if not question:
            continue
        
        print()
        result = ask_bot(question)
        
        print(f"\n🤖 Réponse:")
        print("-" * 80)
        print(result['answer'])
        print("-" * 80)
        
        if result['sources']:
            print(f"\n📚 Basé sur: {', '.join(result['sources'])}")
        
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_ask_bot()
    else:
        test_ask_bot()
