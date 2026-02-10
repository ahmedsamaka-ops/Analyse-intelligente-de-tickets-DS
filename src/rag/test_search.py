"""
Script de test de recherche dans la base vectorielle
"""
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_vector_db(db_path="./simple_vector_db"):
    """
    Charge la base vectorielle depuis le disque
    """
    print(f"📂 Chargement de la base depuis {db_path}...")
    
    with open(f"{db_path}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{db_path}/matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    with open(f"{db_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(f"{db_path}/metadatas.pkl", "rb") as f:
        metadatas = pickle.load(f)
    
    print(f"   ✅ Base chargée: {matrix.shape[0]} documents")
    return vectorizer, matrix, chunks, metadatas


def similarity_search(query, vectorizer, matrix, chunks, metadatas, k=3):
    """
    Recherche de similarité
    
    Args:
        query: La requête de l'utilisateur
        k: Nombre de résultats à retourner
    
    Returns:
        Liste des top k résultats avec leurs scores
    """
    # Vectoriser la query
    query_vec = vectorizer.transform([query])
    
    # Calculer les similarités
    similarities = cosine_similarity(query_vec, matrix)[0]
    
    # Trouver les top k
    top_indices = similarities.argsort()[-k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': chunks[idx],
            'metadata': metadatas[idx],
            'similarity': similarities[idx]
        })
    
    return results


def test_multiple_queries():
    """
    Teste plusieurs requêtes pertinentes
    """
    print("="*70)
    print(" TEST DE RECHERCHE VECTORIELLE")
    print("="*70)
    print()
    
    # Charger la base
    vectorizer, matrix, chunks, metadatas = load_vector_db()
    
    # Liste de requêtes de test
    test_queries = [
        "accès au partage",
        "réinitialisation mot de passe",
        "création projet citrix",
        "demande activation compte",
        "problème téléphonique",
    ]
    
    for query in test_queries:
        print("\n" + "="*70)
        print(f"❓ REQUÊTE: '{query}'")
        print("="*70)
        
        results = similarity_search(query, vectorizer, matrix, chunks, metadatas, k=3)
        
        print(f"\n📊 Top 3 résultats:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{'─'*70}")
            print(f"Résultat {i} (similarité: {result['similarity']:.3f}):")
            print(f"{'─'*70}")
            
            # Extraire les infos clés du texte
            lines = result['text'].split('\n')
            for line in lines[:4]:  # Afficher les 4 premières lignes
                print(line)
            
            print(f"\n📋 Métadonnées:")
            print(f"   • Catégorie: {result['metadata']['categorie']}")
            print(f"   • Urgence: {result['metadata']['urgence']}")
            print(f"   • Type: {result['metadata']['type_ticket']}")
            print()
    
    print("="*70)
    print("✅ Tests terminés!")
    print("="*70)


if __name__ == "__main__":
    test_multiple_queries()
