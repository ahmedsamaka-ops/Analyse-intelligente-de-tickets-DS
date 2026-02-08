# =============================================================================
# Script d'Entraînement V3 - Avec Regroupement des Catégories
# Projet : Analyse Intelligente de Tickets Support
# Objectif : Booster l'accuracy en regroupant 54 catégories → ~12 catégories
# =============================================================================

import pandas as pd
import numpy as np
import re
import joblib
import os
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "data"
MODELS_DIR = "models"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# MAPPING DES CATÉGORIES (54 → 12)
# =============================================================================

MAPPING_CATEGORIES = {
    # ========== 1. COMPTE & ACCÈS (AD, SAP, etc.) ==========
    'Création compte AD': 'Gestion Comptes',
    'Compte AD': 'Gestion Comptes',
    'Compte AD désactivation': 'Gestion Comptes',
    'Compte AD dÃ©sactivation': 'Gestion Comptes',
    'CrÃ©ation compte AD': 'Gestion Comptes',
    'Compte SAP': 'Gestion Comptes',
    'accès SAP': 'Gestion Comptes',
    'MDP SAP': 'Gestion Comptes',
    'MDP CFAO': 'Gestion Comptes',
    'Création compte Lims': 'Gestion Comptes',
    'Création compte HPLC': 'Gestion Comptes',
    'Accès compte e-passport CFAO': 'Gestion Comptes',
    'Activation compte HTDS de UV365': 'Gestion Comptes',
    'accès Windows': 'Gestion Comptes',
    'Installation SAP': 'Gestion Comptes',
    
    # ========== 2. PARTAGE & ACCÈS FICHIERS ==========
    'Accès au partage': 'Partage & Accès',
    'Partage': 'Partage & Accès',
    'accès au dossiers': 'Partage & Accès',
    'Augmentation de taille': 'Partage & Accès',
    
    # ========== 3. RÉSEAU & CONNEXION ==========
    'Connexion internet': 'Réseau & Connexion',
    'Réseau / Connexion internet': 'Réseau & Connexion',
    'Connexion Réseau': 'Réseau & Connexion',
    'VPN': 'Réseau & Connexion',
    'ouverture des Port Veeam': 'Réseau & Connexion',
    
    # ========== 4. IMPRESSIONS & SCANNER ==========
    'Impressions & Scanner/Incident': 'Impressions & Scanner',
    'Impressions & Scanner': 'Impressions & Scanner',
    'Impressions Scanner Request': 'Impressions & Scanner',
    'Accès scanner': 'Impressions & Scanner',
    
    # ========== 5. MATÉRIEL & ÉQUIPEMENT ==========
    'Laptop/Request': 'Matériel & Équipement',
    'Desktop/Request': 'Matériel & Équipement',
    'Matériel/Incident': 'Matériel & Équipement',
    'Affectation PC': 'Matériel & Équipement',
    'Changement de bande': 'Matériel & Équipement',
    'Accessoires/Request': 'Matériel & Équipement',
    'Configuration tél': 'Matériel & Équipement',
    
    # ========== 6. APPLICATIONS & LOGICIELS ==========
    'Applications': 'Applications & Logiciels',
    'Installation TEAMS': 'Applications & Logiciels',
    'Activation office': 'Applications & Logiciels',
    'MAJ system': 'Applications & Logiciels',
    
    # ========== 7. BUREAUTIQUE ==========
    'Bureautique/Incident': 'Bureautique',
    'Bureautique': 'Bureautique',
    'Utilitaires/Request': 'Bureautique',
    'Utilitaires/Incident': 'Bureautique',
    
    # ========== 8. PROJETS ==========
    'Création de projet': 'Gestion Projets',
    'Accès au projet': 'Gestion Projets',
    
    # ========== 9. SÉCURITÉ ==========
    'Sécurité/Sophos': 'Sécurité',
    'débloquer les channels': 'Sécurité',
    
    # ========== 10. SAUVEGARDE & SYSTÈME ==========
    'Sauvegarde': 'Système & Sauvegarde',
    'Système/Incident': 'Système & Sauvegarde',
    'Accès au serveurs': 'Système & Sauvegarde',
    'Statistique SSID': 'Système & Sauvegarde',
    
    # ========== 11. AUTRE ==========
    'Autre': 'Autre',
}

def mapper_categorie(cat):
    """Mappe une catégorie vers sa catégorie regroupée"""
    if pd.isna(cat):
        return 'Autre'
    cat = str(cat).strip()
    return MAPPING_CATEGORIES.get(cat, 'Autre')

# =============================================================================
# FONCTIONS
# =============================================================================

def nettoyer_texte(texte):
    if pd.isna(texte):
        return ""
    texte = str(texte).lower()
    texte = re.sub(r'[^\w\s\-àâäéèêëïîôùûüç]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def charger_donnees():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
    return train_df, test_df, val_df

# =============================================================================
# MAIN
# =============================================================================

print("\n" + "=" * 70)
print("🚀 ENTRAÎNEMENT V3 - REGROUPEMENT DES CATÉGORIES")
print("=" * 70)

# Charger les données
train_df, test_df, val_df = charger_donnees()

# Appliquer le regroupement
print("\n📦 Regroupement des catégories (54 → ~12)...")
train_df['categorie_groupe'] = train_df['categorie'].apply(mapper_categorie)
test_df['categorie_groupe'] = test_df['categorie'].apply(mapper_categorie)
val_df['categorie_groupe'] = val_df['categorie'].apply(mapper_categorie)

# Afficher la distribution
print("\n📊 Nouvelle distribution des catégories :")
distribution = train_df['categorie_groupe'].value_counts()
for cat, count in distribution.items():
    pct = count / len(train_df) * 100
    print(f"   {cat:<25} : {count:3d} ({pct:5.1f}%)")

print(f"\n✅ Nombre de catégories : {len(distribution)} (au lieu de 54)")

# Préparer les données
train_df['texte_clean'] = train_df['texte'].apply(nettoyer_texte)
test_df['texte_clean'] = test_df['texte'].apply(nettoyer_texte)
val_df['texte_clean'] = val_df['texte'].apply(nettoyer_texte)

X_train = train_df['texte_clean']
y_train = train_df['categorie_groupe']
X_test = test_df['texte_clean']
y_test = test_df['categorie_groupe']
X_val = val_df['texte_clean']
y_val = val_df['categorie_groupe']

# Encoder les labels
label_encoder = LabelEncoder()
all_categories = pd.concat([y_train, y_test, y_val]).unique()
label_encoder.fit(all_categories)

y_train_enc = label_encoder.transform(y_train)
y_test_enc = label_encoder.transform(y_test)
y_val_enc = label_encoder.transform(y_val)

# Vectorisation TF-IDF
print("\n🔄 Vectorisation TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
X_val_tfidf = tfidf.transform(X_val)

print(f"✅ Features : {X_train_tfidf.shape[1]}")

# =============================================================================
# COMPARAISON DES MODÈLES
# =============================================================================

print("\n" + "=" * 70)
print("🔬 COMPARAISON DES MODÈLES (avec catégories regroupées)")
print("=" * 70)

modeles = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=3,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'SVM Linear': SVC(
        kernel='linear', C=1.0, probability=True, random_state=RANDOM_STATE
    )
}

meilleur = {'nom': '', 'modele': None, 'acc': 0}

for nom, modele in modeles.items():
    print(f"\n📊 {nom}...")
    
    modele.fit(X_train_tfidf, y_train_enc)
    
    # Test
    y_test_pred = modele.predict(X_test_tfidf)
    acc_test = accuracy_score(y_test_enc, y_test_pred)
    
    # Validation
    y_val_pred = modele.predict(X_val_tfidf)
    acc_val = accuracy_score(y_val_enc, y_val_pred)
    f1_val = f1_score(y_val_enc, y_val_pred, average='weighted', zero_division=0)
    
    # Cross-validation
    cv_scores = cross_val_score(modele, X_train_tfidf, y_train_enc, cv=5)
    
    print(f"   Test       : {acc_test*100:.2f}%")
    print(f"   Validation : {acc_val*100:.2f}%")
    print(f"   F1-Score   : {f1_val*100:.2f}%")
    print(f"   CV (5-fold): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*200:.2f}%)")
    
    if acc_val > meilleur['acc']:
        meilleur = {'nom': nom, 'modele': modele, 'acc': acc_val, 'f1': f1_val}

# =============================================================================
# RÉSULTATS FINAUX
# =============================================================================

print("\n" + "=" * 70)
print("🏆 RÉSULTATS FINAUX")
print("=" * 70)

print(f"\n✅ MEILLEUR MODÈLE : {meilleur['nom']}")
print(f"   Accuracy Validation : {meilleur['acc']*100:.2f}%")
print(f"   F1-Score            : {meilleur['f1']*100:.2f}%")

# Comparaison avec avant
baseline_v1 = 66.09
amelioration = meilleur['acc']*100 - baseline_v1
print(f"\n📈 AMÉLIORATION vs V1 (66.09%) : {amelioration:+.2f}%")

# Rapport de classification
print("\n📋 Rapport de classification détaillé :")
y_val_pred_final = meilleur['modele'].predict(X_val_tfidf)
print(classification_report(
    y_val_enc, y_val_pred_final,
    target_names=label_encoder.classes_,
    zero_division=0
))

# Sauvegarder les modèles
print("\n💾 Sauvegarde des modèles V3...")
joblib.dump(meilleur['modele'], os.path.join(MODELS_DIR, "classification_categorie_model_v3.pkl"))
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer_categorie_v3.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder_categorie_v3.pkl"))

# Sauvegarder aussi le mapping
joblib.dump(MAPPING_CATEGORIES, os.path.join(MODELS_DIR, "mapping_categories.pkl"))

print("✅ Fichiers sauvegardés :")
print("   - classification_categorie_model_v3.pkl")
print("   - tfidf_vectorizer_categorie_v3.pkl")
print("   - label_encoder_categorie_v3.pkl")
print("   - mapping_categories.pkl")

print("\n" + "=" * 70)
print("✅ TERMINÉ !")
print("=" * 70)
