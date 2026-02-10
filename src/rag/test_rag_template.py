"""
Test du template RAG avec contexte
"""
from chatbot import chat_with_context, PROVIDER

def test_rag_template():
    """
    Test du template RAG avec différents contextes bidons
    """
    print("=" * 70)
    print(" TEST DU TEMPLATE RAG")
    print(f" Provider: {PROVIDER}")
    print("=" * 70)
    
    # Test 1: Question dans le contexte
    print("\n📋 TEST 1: Question dont la réponse EST dans le contexte")
    print("-" * 70)
    
    context1 = """
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
    
    question1 = "Combien de temps le lien de réinitialisation est-il valable ?"
    
    print(f"\n📄 Contexte:\n{context1.strip()}")
    print(f"\n❓ Question: {question1}")
    print(f"\n🤖 Réponse attendue: Le lien est valable pendant 24 heures")
    print(f"🤖 Réponse obtenue: ", end="", flush=True)
    
    response1 = chat_with_context(question1, context1)
    print(response1)
    
    # Test 2: Question hors contexte
    print("\n" + "=" * 70)
    print("📋 TEST 2: Question dont la réponse N'EST PAS dans le contexte")
    print("-" * 70)
    
    question2 = "Quelle est la capitale de la France ?"
    
    print(f"\n📄 Contexte: (le même document sur les mots de passe)")
    print(f"\n❓ Question: {question2}")
    print(f"\n🤖 Réponse attendue: Je ne trouve pas cette information dans les documents.")
    print(f"🤖 Réponse obtenue: ", end="", flush=True)
    
    response2 = chat_with_context(question2, context1)
    print(response2)
    
    # Test 3: Contexte technique Maroc
    print("\n" + "=" * 70)
    print("📋 TEST 3: Contexte technique avec expressions marocaines")
    print("-" * 70)
    
    context3 = """
    Ticket #12345 - Problème de connexion VPN
    
    Client: Entreprise TechMaroc à Casablanca
    Problème: Le VPN ma kaykhdamch (ne fonctionne pas)
    
    Solution appliquée:
    - Vérifié les paramètres réseau
    - Réinstallé le certificat VPN
    - Redémarré le service VPN
    - Temps de résolution: 2 heures
    
    Statut: Résolu - Le client peut maintenant se connecter sans problème.
    """
    
    question3 = "Combien de temps a pris la résolution du problème ?"
    
    print(f"\n📄 Contexte:\n{context3.strip()}")
    print(f"\n❓ Question: {question3}")
    print(f"\n🤖 Réponse attendue: 2 heures")
    print(f"🤖 Réponse obtenue: ", end="", flush=True)
    
    response3 = chat_with_context(question3, context3)
    print(response3)
    
    # Test 4: Information partielle
    print("\n" + "=" * 70)
    print("📋 TEST 4: Question sur info non mentionnée")
    print("-" * 70)
    
    question4 = "Quel est le prix du service ?"
    
    print(f"\n📄 Contexte: (le même ticket VPN)")
    print(f"\n❓ Question: {question4}")
    print(f"\n🤖 Réponse attendue: Je ne trouve pas cette information...")
    print(f"🤖 Réponse obtenue: ", end="", flush=True)
    
    response4 = chat_with_context(question4, context3)
    print(response4)
    
    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    tests_results = [
        ("Info présente (délai 24h)", "24" in response1.lower() or "heures" in response1.lower()),
        ("Hors contexte (capitale)", "ne trouve pas" in response2.lower() or "ne sais pas" in response2.lower()),
        ("Info présente (2 heures)", "2" in response3 and "heure" in response3.lower()),
        ("Info absente (prix)", "ne trouve pas" in response4.lower() or "ne sais pas" in response4.lower()),
    ]
    
    print()
    for test_name, passed in tests_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in tests_results if passed)
    print(f"\n🎯 Score: {total_passed}/{len(tests_results)} tests passés")
    
    if total_passed == len(tests_results):
        print("\n✨ Parfait ! Le template RAG fonctionne correctement.")
    elif total_passed >= len(tests_results) / 2:
        print("\n⚠️ Le template fonctionne mais peut être amélioré.")
    else:
        print("\n❌ Le template a besoin d'ajustements.")
    
    print("=" * 70)
    
    # Sauvegarder les résultats
    with open("rag_template_test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Tests du template RAG\n")
        f.write(f"Provider: {PROVIDER}\n")
        f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, ((test_name, passed), question, response) in enumerate(zip(
            tests_results,
            [question1, question2, question3, question4],
            [response1, response2, response3, response4]
        ), 1):
            status = "PASS" if passed else "FAIL"
            f.write(f"\n{'='*60}\n")
            f.write(f"Test {i}: {test_name} - {status}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Réponse: {response}\n")
        
        f.write(f"\n\nScore final: {total_passed}/{len(tests_results)}\n")
    
    print("\n📝 Résultats détaillés sauvegardés dans: rag_template_test_results.txt")


if __name__ == "__main__":
    test_rag_template()
