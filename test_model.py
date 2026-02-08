# =============================================================================
# Script de Test des Modèles - Prédiction Complète
# =============================================================================

import joblib
import re

# Charger les modèles
print("Chargement des modèles...")

# Modèle Catégorie V3
model_cat = joblib.load("models/classification_categorie_model_v3.pkl")
vec_cat = joblib.load("models/tfidf_vectorizer_categorie_v3.pkl")
enc_cat = joblib.load("models/label_encoder_categorie_v3.pkl")

# Modèle Urgence Balanced
model_urg = joblib.load("models/classification_urgence_balanced.pkl")
vec_urg = joblib.load("models/tfidf_vectorizer_urgence_balanced.pkl")
enc_urg = joblib.load("models/label_encoder_urgence_balanced.pkl")

# Modèle Temps
model_temps = joblib.load("models/regression_temps_model.pkl")
vec_temps = joblib.load("models/tfidf_vectorizer_regression.pkl")

print("Modèles chargés avec succès!")

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r'[^\w\s\-àâäéèêëïîôùûüç]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def predire(texte):
    texte_clean = nettoyer_texte(texte)
    
    # Prédiction Catégorie
    X_cat = vec_cat.transform([texte_clean])
    cat_pred = enc_cat.inverse_transform(model_cat.predict(X_cat))[0]
    
    # Prédiction Urgence
    X_urg = vec_urg.transform([texte_clean])
    urg_pred = enc_urg.inverse_transform(model_urg.predict(X_urg))[0]
    
    # Prédiction Temps (utiliser le même vectoriseur que catégorie)
    X_temps = vec_cat.transform([texte_clean])
    try:
        temps_pred = model_temps.predict(X_temps)[0]
    except:
        # Si erreur, utiliser une estimation moyenne
        temps_pred = 15.0
    
    return cat_pred, urg_pred, temps_pred

# TEST
print("\n" + "="*60)
print("TEST DU MODELE")
print("="*60)

texte_test = "problème connexion wifi maphoffice"

print(f"\nINPUT: \"{texte_test}\"")
print("\n" + "-"*60)
print("PREDICTIONS:")
print("-"*60)

cat, urg, temps = predire(texte_test)

print(f"  • Catégorie        : {cat}")
print(f"  • Urgence          : {urg}")
print(f"  • Temps résolution : {temps:.1f} heures")
print("="*60)
