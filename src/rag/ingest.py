"""
Script d'ingestion des données dans la base vectorielle
Utilise SentenceTransformers directement pour les embeddings
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_and_chunk_tickets(csv_path: str = "data/tickets.csv", max_chunks: int = 100):
    """
    Charge les tickets depuis le CSV et les prépare en chunks
    
    Args:
        csv_path: Chemin vers le fichier CSV des tickets
        max_chunks: Nombre maximum de chunks à créer (pour test rapide)
    
    Returns:
        Liste de textes (chunks) et leurs métadonnées
    """
    print(f"📂 Chargement des tickets depuis {csv_path}...")
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    print(f"   ✅ {len(df)} tickets chargés")
    
    # Préparer les chunks (texte complet du ticket)
    chunks = []
    metadatas = []
    
    for idx, row in df.head(max_chunks).iterrows():
        # Créer un texte complet pour chaque ticket
        text = f"""Ticket ID: {row['ID']}
Titre: {row['titre']}
Catégorie: {row['categorie']}
Urgence: {row['urgence']}
Type: {row['type_ticket']}
Temps de résolution: {row['temps_resolution']} heures

Texte: {row['text_full']}"""
        
        chunks.append(text)
        metadatas.append({
            "id": str(row['ID']),
            "categorie": row['categorie'],
            "urgence": row['urgence'],
            "type_ticket": row['type_ticket']
        })
    
    print(f"   ✅ {len(chunks)} chunks créés")
    return chunks, metadatas


def test_embeddings_with_langchain():
    """
    Test du modèle d'embeddings avec LangChain
    Note: Nécessite l'installation de langchain-community
    """
    print("🔧 Test avec LangChain (langchain_community.embeddings)...")
    
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        
        # Charger le modèle
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("✅ LangChain chargé avec succès!")
        
        # Test avec un mot simple
        print("\n📝 Test d'embedding sur le mot 'bonjour'...")
        result = embeddings.embed_query("bonjour")
        
        print(f"\n📊 Résultat (vecteur de {len(result)} dimensions):")
        print(result[:10])  # Afficher les 10 premières valeurs
        print("...")
        print(result[-10:])  # Afficher les 10 dernières valeurs
        
        print(f"\n✅ Le modèle fonctionne correctement avec LangChain!")
        print(f"   Dimension du vecteur: {len(result)}")
        print(f"   Type: {type(result)}")
        
        return embeddings
    
    except ImportError as e:
        print(f"❌ Erreur d'import LangChain: {e}")
        print("   LangChain nécessite des dépendances avec Rust compiler")
        print("   Essai avec sentence-transformers directement...\n")
        return None


def test_embeddings_direct():
    """
    Test du modèle d'embeddings directement avec sentence-transformers
    Alternative qui fonctionne sans LangChain
    """
    print("🔧 Chargement du modèle d'embeddings (sentence-transformers)...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Charger le modèle
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("✅ Modèle chargé avec succès!")
        
        # Test avec un mot simple
        print("\n📝 Test d'embedding sur le mot 'bonjour'...")
        result = model.encode("bonjour")
        
        print(f"\n📊 Résultat (vecteur de {len(result)} dimensions):")
        print(result[:10])  # Afficher les 10 premières valeurs
        print("...")
        print(result[-10:])  # Afficher les 10 dernières valeurs
        
        print(f"\n✅ Le modèle fonctionne correctement!")
        print(f"   Dimension du vecteur: {len(result)}")
        print(f"   Type: {type(result)}")
        
        # Test avec une phrase
        print("\n" + "="*60)
        print("📝 Test avec une phrase complète...")
        phrase = "Comment résoudre un problème de connexion Maroc Telecom?"
        result2 = model.encode(phrase)
        
        print(f"Phrase: {phrase}")
        print(f"Vecteur de {len(result2)} dimensions généré")
        print(f"Premiers éléments: {result2[:5]}")
        
        return model
    
    except ImportError as e:
        print(f"❌ Erreur: {e}")
        print("   Installez sentence-transformers: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None


def create_chroma_db_with_langchain(chunks, metadatas):
    """
    Crée une base Chroma avec LangChain
    """
    print("\n" + "="*60)
    print("🗄️ CRÉATION DE LA BASE CHROMA (LangChain)")
    print("="*60)
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        
        # Charger le modèle d'embeddings
        print("\n📦 Chargement du modèle d'embeddings...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("   ✅ Modèle chargé")
        
        # Créer le dossier si nécessaire
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        
        # Créer la base Chroma
        print(f"\n🔨 Création de la base Chroma avec {len(chunks)} documents...")
        db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=persist_directory
        )
        
        print(f"   ✅ Base créée dans {persist_directory}")
        
        # Test de recherche
        print("\n" + "="*60)
        print("🔍 TEST DE RECHERCHE DE SIMILARITÉ")
        print("="*60)
        
        query = "probleme wifi"
        print(f"\n❓ Requête: '{query}'")
        print(f"📊 Recherche des 3 documents les plus similaires...")
        
        results = db.similarity_search(query, k=3)
        
        print(f"\n✅ {len(results)} résultats trouvés:\n")
        
        for i, doc in enumerate(results, 1):
            print(f"{'='*60}")
            print(f"Résultat {i}:")
            print(f"{'='*60}")
            print(doc.page_content[:200] + "...")
            if doc.metadata:
                print(f"\nMétadonnées: {doc.metadata}")
            print()
        
        print("="*60)
        print("✅ La base Chroma fonctionne correctement!")
        print(f"📂 Dossier créé: {persist_directory}")
        print("="*60)
        
        return db
    
    except ImportError as e:
        print(f"\n❌ Erreur d'import: {e}")
        print("   Packages requis: langchain-community, chromadb, sentence-transformers")
        print("   Ces packages nécessitent un compilateur Rust sur Python 3.14")
        return None
    except Exception as e:
        print(f"\n❌ Erreur lors de la création: {e}")
        return None


def create_simple_vector_db(chunks, metadatas):
    """
    Crée une base vectorielle simple avec scikit-learn (TF-IDF)
    Alternative qui fonctionne sans dépendances complexes
    """
    print("\n" + "="*60)
    print("🗄️ CRÉATION DE LA BASE VECTORIELLE (TF-IDF)")
    print("="*60)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import pickle
        
        # Créer le vectorizer TF-IDF
        print("\n📦 Création du vectorizer TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        
        # Vectoriser les chunks
        print(f"🔨 Vectorisation de {len(chunks)} documents...")
        tfidf_matrix = vectorizer.fit_transform(chunks)
        print(f"   ✅ Matrice TF-IDF créée: {tfidf_matrix.shape}")
        
        # Sauvegarder
        db_path = "./simple_vector_db"
        os.makedirs(db_path, exist_ok=True)
        
        with open(f"{db_path}/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open(f"{db_path}/matrix.pkl", "wb") as f:
            pickle.dump(tfidf_matrix, f)
        with open(f"{db_path}/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        with open(f"{db_path}/metadatas.pkl", "wb") as f:
            pickle.dump(metadatas, f)
        
        print(f"   ✅ Base sauvegardée dans {db_path}")
        
        # Test de recherche
        print("\n" + "="*60)
        print("🔍 TEST DE RECHERCHE DE SIMILARITÉ")
        print("="*60)
        
        query = "probleme wifi"
        print(f"\n❓ Requête: '{query}'")
        
        # Vectoriser la query
        query_vec = vectorizer.transform([query])
        
        # Calculer les similarités
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        
        # Trouver les top 3
        top_indices = similarities.argsort()[-3:][::-1]
        
        print(f"📊 Top 3 résultats les plus similaires:\n")
        
        for i, idx in enumerate(top_indices, 1):
            print(f"{'='*60}")
            print(f"Résultat {i} (similarité: {similarities[idx]:.3f}):")
            print(f"{'='*60}")
            print(chunks[idx][:200] + "...")
            if metadatas[idx]:
                print(f"\nMétadonnées: {metadatas[idx]}")
            print()
        
        print("="*60)
        print("✅ La base vectorielle TF-IDF fonctionne correctement!")
        print(f"📂 Dossier créé: {db_path}")
        print("="*60)
        
        return {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'chunks': chunks,
            'metadatas': metadatas
        }
    
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return None


if __name__ == "__main__":
    print("="*60)
    print(" INGESTION DES DONNÉES DANS LA BASE VECTORIELLE")
    print("="*60)
    print()
    
    # Étape 1: Charger et chunker les tickets
    chunks, metadatas = load_and_chunk_tickets(max_chunks=50)
    
    # Étape 2: Essayer d'abord avec LangChain
    result_langchain = test_embeddings_with_langchain()
    
    if result_langchain is not None:
        # Si LangChain fonctionne, créer la base Chroma
        db = create_chroma_db_with_langchain(chunks, metadatas)
    else:
        # Sinon, utiliser la solution simple TF-IDF
        print("\n⚠️ LangChain/Chroma non disponible")
        print("   Utilisation de la solution alternative (TF-IDF)...\n")
        db = create_simple_vector_db(chunks, metadatas)
    
    if db is not None:
        print("\n" + "="*60)
        print("✅ SUCCÈS - Base vectorielle créée et testée!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ ÉCHEC - Impossible de créer la base vectorielle")
        print("="*60)
