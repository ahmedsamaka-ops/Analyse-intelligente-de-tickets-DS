"""
Initialisation de la base de données vectorielle ChromaDB
avec des tickets de support pour tests
"""
import chromadb
from chromadb.config import Settings
import os

# Chemin de la base de données
DB_PATH = "data/chroma_db"

# Documents de support (contexte marocain)
SUPPORT_DOCUMENTS = [
    {
        "id": "doc1",
        "text": """Ticket #001 - Problème de connexion Maroc Telecom
        
Client: Entreprise TechCasa à Casablanca
Problème: La connexion Internet ADSL ne fonctionne pas depuis ce matin.

Solution:
1. Vérifier que le modem est bien allumé (voyant vert)
2. Redémarrer le modem (débrancher 30 secondes puis rebrancher)
3. Vérifier les câbles RJ45 et RJ11
4. Si le problème persiste, appeler le 888 (service client Maroc Telecom)
5. Temps de résolution moyen: 2 heures

La connexion a été rétablie après redémarrage du modem.
Statut: Résolu
""",
        "metadata": {"type": "connexion", "operateur": "maroc_telecom", "categorie": "reseau"}
    },
    {
        "id": "doc2",
        "text": """Ticket #002 - Réinitialisation mot de passe

Client: Mohamed Ben Ali - Rabat
Problème: L'utilisateur a oublié son mot de passe et ne peut plus se connecter à l'application.

Procédure de réinitialisation:
1. Aller sur la page de connexion
2. Cliquer sur "Mot de passe oublié"
3. Entrer l'email professionnel (@entreprise.ma)
4. Vérifier l'email (y compris spam)
5. Cliquer sur le lien reçu (valable 24 heures)
6. Créer un nouveau mot de passe (min 8 caractères, 1 majuscule, 1 chiffre)

Si pas d'email reçu: Contacter le support au +212-5XX-XXXXXX
Temps de résolution: 15 minutes
Statut: Résolu
""",
        "metadata": {"type": "authentification", "categorie": "compte_utilisateur"}
    },
    {
        "id": "doc3",
        "text": """Ticket #003 - Problème VPN Orange Maroc

Client: Société DataPro - Tanger
Problème: Le VPN ma kaykhdemch (ne marche pas). Erreur "Connection timeout".

Causes possibles:
- Pare-feu bloquant le port VPN (1194 ou 443)
- Certificat VPN expiré
- Mauvaise configuration DNS

Solution appliquée:
1. Désactiver temporairement le pare-feu Windows
2. Télécharger le nouveau certificat VPN depuis le portail
3. Importer le certificat (double-clic puis suivre l'assistant)
4. Configurer les DNS: 8.8.8.8 et 8.8.4.4
5. Se reconnecter au VPN

Temps de résolution: 3 heures
Statut: Résolu - Le client peut maintenant se connecter sans problème
""",
        "metadata": {"type": "vpn", "operateur": "orange", "categorie": "reseau"}
    },
    {
        "id": "doc4",
        "text": """Ticket #004 - Problème imprimante réseau

Client: Cabinet Comptable - Casablanca
Problème: L'imprimante réseau HP n'imprime plus. Message "Imprimante hors ligne".

Diagnostic:
- L'imprimante est bien connectée au réseau
- L'adresse IP n'a pas changé (192.168.1.50)
- Le problème vient du driver obsolète

Solution:
1. Désinstaller l'ancien driver d'imprimante
2. Télécharger le dernier driver depuis hp.com/maroc
3. Installer le nouveau driver
4. Redémarrer l'ordinateur
5. Ajouter l'imprimante par son IP: 192.168.1.50
6. Faire un test d'impression

Temps de résolution: 1 heure
Statut: Résolu
""",
        "metadata": {"type": "materiel", "categorie": "imprimante"}
    },
    {
        "id": "doc5",
        "text": """Ticket #005 - Lenteur connexion Inwi

Client: Start-up TechInov - Marrakech
Problème: Internet 4G très lent depuis 2 jours. Vitesse < 1 Mbps au lieu de 20 Mbps.

Analyses effectuées:
- Signal 4G correct (3-4 barres)
- Pas de dépassement de quota data
- Problème de congestion réseau dans la zone

Solutions proposées:
1. Redémarrer le routeur 4G
2. Changer de bande de fréquence (passer en manuel sur 4G uniquement)
3. Réinitialiser les paramètres APN:
   - APN: www.inwi.ma
   - Proxy: vide
   - Port: vide
4. Contacter Inwi au 121 pour signaler la lenteur dans la zone
5. Alternative temporaire: Utiliser un VPN pour optimiser le routage

Temps de résolution: 4 heures (avec intervention Inwi)
Statut: Résolu partiellement - Vitesse améliorée à 10 Mbps
""",
        "metadata": {"type": "connexion", "operateur": "inwi", "categorie": "reseau"}
    },
    {
        "id": "doc6",
        "text": """Ticket #006 - Problème email Outlook

Client: Directeur Commercial - Fès
Problème: Ne peut plus envoyer d'emails. Reçoit l'erreur "L'envoi a échoué".

Diagnostic:
- La boîte de réception fonctionne (IMAP OK)
- Le problème est sur le serveur d'envoi (SMTP)

Configuration SMTP correcte:
- Serveur: smtp.office365.com
- Port: 587
- Sécurité: STARTTLS
- Authentification: Oui (même identifiants que IMAP)

Solution:
1. Ouvrir Outlook > Paramètres > Comptes
2. Vérifier les paramètres du serveur sortant
3. Corriger le port (était 25, doit être 587)
4. Activer "Mon serveur sortant requiert une authentification"
5. Tester l'envoi d'un email

Temps de résolution: 20 minutes
Statut: Résolu
""",
        "metadata": {"type": "email", "categorie": "messagerie"}
    }
]


def init_chroma_db():
    """
    Initialise la base de données vectorielle ChromaDB avec des documents de support
    """
    print("🔧 Initialisation de ChromaDB...")
    
    # Créer le dossier si nécessaire
    os.makedirs(DB_PATH, exist_ok=True)
    
    # Initialiser ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Supprimer la collection si elle existe déjà (pour réinitialisation)
    try:
        client.delete_collection("support_tickets")
        print("   ↪ Collection existante supprimée")
    except:
        pass
    
    # Créer une nouvelle collection
    collection = client.create_collection(
        name="support_tickets",
        metadata={"description": "Tickets de support technique - Contexte marocain"}
    )
    
    print(f"   ↪ Collection 'support_tickets' créée")
    
    # Ajouter les documents
    documents = [doc["text"] for doc in SUPPORT_DOCUMENTS]
    ids = [doc["id"] for doc in SUPPORT_DOCUMENTS]
    metadatas = [doc["metadata"] for doc in SUPPORT_DOCUMENTS]
    
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"   ↪ {len(SUPPORT_DOCUMENTS)} documents ajoutés")
    print(f"\n✅ Base de données ChromaDB initialisée avec succès !")
    print(f"📂 Emplacement: {DB_PATH}")
    
    # Afficher un résumé
    print(f"\n📊 Contenu de la base:")
    for doc in SUPPORT_DOCUMENTS:
        print(f"   • {doc['id']}: {doc['metadata'].get('type', 'N/A')} - {doc['metadata'].get('categorie', 'N/A')}")
    
    return collection


def test_search():
    """
    Test rapide de recherche dans la base
    """
    print("\n" + "="*70)
    print("🔍 TEST DE RECHERCHE")
    print("="*70)
    
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("support_tickets")
    
    test_queries = [
        "Comment résoudre un problème de connexion Maroc Telecom?",
        "Réinitialiser mot de passe oublié",
        "VPN ne fonctionne pas",
        "Imprimante ne marche pas"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        results = collection.query(
            query_texts=[query],
            n_results=2
        )
        
        print(f"   📄 Top 2 résultats:")
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0]), 1):
            print(f"      {i}. {doc_id} (distance: {distance:.3f})")


if __name__ == "__main__":
    init_chroma_db()
    test_search()
