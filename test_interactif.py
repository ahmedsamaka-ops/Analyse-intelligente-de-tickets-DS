# =============================================================================
# Script de Test Interactif - Tester plusieurs tickets
# =============================================================================

import joblib
import re

# Charger les modèles
print("Chargement des modèles...")
model_cat = joblib.load("models/classification_categorie_model_v3.pkl")
vec_cat = joblib.load("models/tfidf_vectorizer_categorie_v3.pkl")
enc_cat = joblib.load("models/label_encoder_categorie_v3.pkl")
model_urg = joblib.load("models/classification_urgence_balanced.pkl")
vec_urg = joblib.load("models/tfidf_vectorizer_urgence_balanced.pkl")
enc_urg = joblib.load("models/label_encoder_urgence_balanced.pkl")
print("OK!\n")

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r'[^\w\s\-àâäéèêëïîôùûüç]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def predire(texte):
    texte_clean = nettoyer_texte(texte)
    X_cat = vec_cat.transform([texte_clean])
    X_urg = vec_urg.transform([texte_clean])
    cat = enc_cat.inverse_transform(model_cat.predict(X_cat))[0]
    urg = enc_urg.inverse_transform(model_urg.predict(X_urg))[0]
    return cat, urg

print("="*60)
print("   TEST INTERACTIF DES MODELES")
print("   Tape 'quit' pour quitter")
print("="*60)

while True:
    print()
    texte = input("Tape ton ticket : ")
    
    if texte.lower() in ['quit', 'exit', 'q']:
        print("Au revoir!")
        break
    
    if texte.strip() == "":
        continue
    
    cat, urg = predire(texte)
    
    print(f"\n  → Catégorie : {cat}")
    print(f"  → Urgence   : {urg}")
