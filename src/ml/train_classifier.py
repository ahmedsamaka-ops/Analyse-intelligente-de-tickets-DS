# =============================================================================
# Script d'Entraînement du Modèle de Classification
# Projet : Analyse Intelligente de Tickets Support
# Auteur : Expert ML
# Date : Février 2026
# =============================================================================
# Ce script entraîne deux modèles de classification :
# 1. Classification de la CATÉGORIE du ticket
# 2. Classification de l'URGENCE du ticket
# =============================================================================

import pandas as pd
import numpy as np
import re
import joblib
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Chemins des fichiers
DATA_DIR = "data"
MODELS_DIR = "models"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")

# Créer le dossier models s'il n'existe pas
os.makedirs(MODELS_DIR, exist_ok=True)

# Seed pour la reproductibilité
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# 1. FONCTIONS DE PRÉTRAITEMENT
# =============================================================================

def nettoyer_texte(texte):
    """
    Nettoie un texte pour l'analyse ML.
    
    Args:
        texte: Le texte à nettoyer
        
    Returns:
        Le texte nettoyé
    """
    if pd.isna(texte):
        return ""
    
    texte = str(texte)
    
    # Corrections d'encodage
    corrections = {
        'Ã©': 'é', 'Ã¨': 'è', 'Ãª': 'ê', 'Ã ': 'à',
        'Ã§': 'ç', 'Ã´': 'ô', 'Ã®': 'î', 'Ã¯': 'ï',
        'Ã¹': 'ù', 'Ã»': 'û', 'Ã¢': 'â', 'â€™': "'",
    }
    for ancien, nouveau in corrections.items():
        texte = texte.replace(ancien, nouveau)
    
    # Minuscules
    texte = texte.lower()
    
    # Supprimer caractères spéciaux
    texte = re.sub(r'[^\w\s\-àâäéèêëïîôùûüç]', ' ', texte)
    
    # Supprimer espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte


def charger_donnees():
    """
    Charge les données d'entraînement, test et validation.
    
    Returns:
        Tuple de DataFrames (train, test, validation)
    """
    print("=" * 70)
    print("📂 CHARGEMENT DES DONNÉES")
    print("=" * 70)
    
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)
    
    print(f"✅ Train     : {len(train_df):,} tickets")
    print(f"✅ Test      : {len(test_df):,} tickets")
    print(f"✅ Validation: {len(val_df):,} tickets")
    print(f"📊 Total     : {len(train_df) + len(test_df) + len(val_df):,} tickets")
    
    return train_df, test_df, val_df


def preparer_donnees(df, colonne_texte='texte'):
    """
    Prépare les données pour l'entraînement.
    
    Args:
        df: DataFrame avec les données
        colonne_texte: Nom de la colonne contenant le texte
        
    Returns:
        DataFrame avec le texte nettoyé
    """
    df = df.copy()
    df['texte_clean'] = df[colonne_texte].apply(nettoyer_texte)
    return df


# =============================================================================
# 2. ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION DE CATÉGORIE
# =============================================================================

def entrainer_classificateur_categorie(train_df, test_df, val_df):
    """
    Entraîne un modèle de classification pour prédire la catégorie du ticket.
    
    Returns:
        Dictionnaire avec le modèle, le vectoriseur, l'encodeur et les métriques
    """
    print("\n" + "=" * 70)
    print("🏷️  CLASSIFICATION DES CATÉGORIES")
    print("=" * 70)
    
    # Préparer les données
    train_df = preparer_donnees(train_df)
    test_df = preparer_donnees(test_df)
    val_df = preparer_donnees(val_df)
    
    # Textes et labels
    X_train = train_df['texte_clean']
    y_train = train_df['categorie']
    X_test = test_df['texte_clean']
    y_test = test_df['categorie']
    X_val = val_df['texte_clean']
    y_val = val_df['categorie']
    
    # Encoder les labels
    label_encoder = LabelEncoder()
    
    # Combiner toutes les catégories pour s'assurer que l'encodeur connaît toutes les classes
    all_categories = pd.concat([y_train, y_test, y_val]).unique()
    label_encoder.fit(all_categories)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)
    
    print(f"\n📋 Nombre de catégories : {len(label_encoder.classes_)}")
    print(f"📋 Catégories : {list(label_encoder.classes_[:10])}...")
    
    # Vectorisation TF-IDF
    print("\n🔄 Vectorisation TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigrammes et bigrammes
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    
    print(f"✅ Vocabulaire : {len(tfidf_vectorizer.vocabulary_):,} termes")
    print(f"✅ Shape train : {X_train_tfidf.shape}")
    
    # Comparaison de plusieurs modèles
    print("\n🔬 Comparaison des modèles...")
    modeles = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1)
    }
    
    meilleurs_resultats = {'modele': None, 'nom': '', 'accuracy': 0}
    
    for nom, modele in modeles.items():
        print(f"\n   📊 {nom}...")
        modele.fit(X_train_tfidf, y_train_encoded)
        y_pred = modele.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"      Accuracy Test : {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy > meilleurs_resultats['accuracy']:
            meilleurs_resultats = {
                'modele': modele,
                'nom': nom,
                'accuracy': accuracy
            }
    
    # Utiliser le meilleur modèle
    meilleur_modele = meilleurs_resultats['modele']
    print(f"\n🏆 Meilleur modèle : {meilleurs_resultats['nom']}")
    
    # Évaluation finale sur le set de validation
    print("\n" + "-" * 50)
    print("📈 ÉVALUATION FINALE (Set de Validation)")
    print("-" * 50)
    
    y_val_pred = meilleur_modele.predict(X_val_tfidf)
    
    # Métriques
    accuracy = accuracy_score(y_val_encoded, y_val_pred)
    precision = precision_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    
    print(f"\n   🎯 Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📊 Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"   📊 Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"   📊 F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    
    # Cross-validation sur l'ensemble train
    print("\n   🔄 Cross-Validation (5-fold) sur Train...")
    cv_scores = cross_val_score(meilleur_modele, X_train_tfidf, y_train_encoded, cv=5)
    print(f"      Scores CV : {cv_scores}")
    print(f"      Moyenne   : {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Sauvegarder les modèles
    print("\n💾 Sauvegarde des modèles...")
    
    modele_path = os.path.join(MODELS_DIR, "classification_categorie_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer_categorie.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder_categorie.pkl")
    
    joblib.dump(meilleur_modele, modele_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"   ✅ Modèle     : {modele_path}")
    print(f"   ✅ Vectorizer : {vectorizer_path}")
    print(f"   ✅ Encoder    : {encoder_path}")
    
    return {
        'modele': meilleur_modele,
        'vectorizer': tfidf_vectorizer,
        'encoder': label_encoder,
        'metriques': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        },
        'nom_modele': meilleurs_resultats['nom']
    }


# =============================================================================
# 3. ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION D'URGENCE
# =============================================================================

def entrainer_classificateur_urgence(train_df, test_df, val_df):
    """
    Entraîne un modèle de classification pour prédire l'urgence du ticket.
    
    Returns:
        Dictionnaire avec le modèle, le vectoriseur, l'encodeur et les métriques
    """
    print("\n" + "=" * 70)
    print("⚡ CLASSIFICATION DE L'URGENCE")
    print("=" * 70)
    
    # Préparer les données
    train_df = preparer_donnees(train_df)
    test_df = preparer_donnees(test_df)
    val_df = preparer_donnees(val_df)
    
    # Textes et labels
    X_train = train_df['texte_clean']
    y_train = train_df['urgence']
    X_test = test_df['texte_clean']
    y_test = test_df['urgence']
    X_val = val_df['texte_clean']
    y_val = val_df['urgence']
    
    print(f"\n📋 Distribution des urgences (Train) :")
    for urgence, count in y_train.value_counts().items():
        print(f"   - {urgence}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Encoder les labels - combiner toutes les urgences
    label_encoder = LabelEncoder()
    all_urgences = pd.concat([y_train, y_test, y_val]).unique()
    label_encoder.fit(all_urgences)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Vectorisation TF-IDF
    print("\n🔄 Vectorisation TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    
    # Entraînement avec Random Forest (bon pour les classes déséquilibrées)
    print("\n🔬 Entraînement du modèle...")
    modele = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',  # Important pour les classes déséquilibrées
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    modele.fit(X_train_tfidf, y_train_encoded)
    
    # Évaluation
    print("\n" + "-" * 50)
    print("📈 ÉVALUATION FINALE (Set de Validation)")
    print("-" * 50)
    
    y_val_pred = modele.predict(X_val_tfidf)
    
    # Métriques
    accuracy = accuracy_score(y_val_encoded, y_val_pred)
    precision = precision_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val_encoded, y_val_pred, average='weighted', zero_division=0)
    
    print(f"\n   🎯 Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📊 Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"   📊 Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"   📊 F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    
    # Rapport de classification détaillé
    print("\n📋 Rapport de classification :")
    # Obtenir les classes présentes dans y_val
    classes_presentes = np.unique(np.concatenate([y_val_encoded, y_val_pred]))
    target_names_presentes = [label_encoder.classes_[i] for i in classes_presentes]
    print(classification_report(
        y_val_encoded, y_val_pred,
        labels=classes_presentes,
        target_names=target_names_presentes,
        zero_division=0
    ))
    
    # Sauvegarder les modèles
    print("💾 Sauvegarde des modèles...")
    
    modele_path = os.path.join(MODELS_DIR, "classification_urgence_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer_urgence.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder_urgence.pkl")
    
    joblib.dump(modele, modele_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"   ✅ Modèle     : {modele_path}")
    print(f"   ✅ Vectorizer : {vectorizer_path}")
    print(f"   ✅ Encoder    : {encoder_path}")
    
    return {
        'modele': modele,
        'vectorizer': tfidf_vectorizer,
        'encoder': label_encoder,
        'metriques': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    }


# =============================================================================
# 4. FONCTION DE PRÉDICTION
# =============================================================================

def predire_ticket(texte, type_prediction='categorie'):
    """
    Prédit la catégorie ou l'urgence d'un ticket.
    
    Args:
        texte: Le texte du ticket
        type_prediction: 'categorie' ou 'urgence'
        
    Returns:
        Dictionnaire avec la prédiction et les probabilités
    """
    # Charger les modèles
    if type_prediction == 'categorie':
        modele = joblib.load(os.path.join(MODELS_DIR, "classification_categorie_model.pkl"))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_categorie.pkl"))
        encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder_categorie.pkl"))
    else:
        modele = joblib.load(os.path.join(MODELS_DIR, "classification_urgence_model.pkl"))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_urgence.pkl"))
        encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder_urgence.pkl"))
    
    # Nettoyer et vectoriser le texte
    texte_clean = nettoyer_texte(texte)
    texte_tfidf = vectorizer.transform([texte_clean])
    
    # Prédiction
    prediction_encoded = modele.predict(texte_tfidf)[0]
    prediction = encoder.inverse_transform([prediction_encoded])[0]
    
    # Probabilités
    probas = modele.predict_proba(texte_tfidf)[0]
    probas_dict = {
        encoder.classes_[i]: float(prob) 
        for i, prob in enumerate(probas)
    }
    
    # Top 3 des catégories les plus probables
    top_3 = sorted(probas_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        'prediction': prediction,
        'confiance': float(max(probas)),
        'probabilites': probas_dict,
        'top_3': top_3
    }


# =============================================================================
# 5. MAIN - EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION")
    print("   Projet : Analyse Intelligente de Tickets Support")
    print("   Date   :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Charger les données
    train_df, test_df, val_df = charger_donnees()
    
    # 1. Classification des catégories
    resultats_categorie = entrainer_classificateur_categorie(train_df, test_df, val_df)
    
    # 2. Classification de l'urgence
    resultats_urgence = entrainer_classificateur_urgence(train_df, test_df, val_df)
    
    # Résumé final
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ FINAL - MODÈLES DE CLASSIFICATION")
    print("=" * 70)
    
    print("\n🏷️  Classification CATÉGORIE :")
    print(f"   - Modèle utilisé : {resultats_categorie['nom_modele']}")
    print(f"   - Accuracy       : {resultats_categorie['metriques']['accuracy']*100:.2f}%")
    print(f"   - F1-Score       : {resultats_categorie['metriques']['f1_score']*100:.2f}%")
    print(f"   - CV Mean        : {resultats_categorie['metriques']['cv_mean']*100:.2f}%")
    
    print("\n⚡ Classification URGENCE :")
    print(f"   - Accuracy       : {resultats_urgence['metriques']['accuracy']*100:.2f}%")
    print(f"   - F1-Score       : {resultats_urgence['metriques']['f1_score']*100:.2f}%")
    
    print("\n📁 Fichiers sauvegardés dans le dossier 'models/':")
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pkl'):
            size = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
            print(f"   - {f} ({size:.1f} KB)")
    
    # Test avec un exemple
    print("\n" + "=" * 70)
    print("🧪 TEST AVEC UN EXEMPLE")
    print("=" * 70)
    
    exemple_texte = "problème de connexion wifi impossible de se connecter au réseau"
    print(f"\n📝 Texte : \"{exemple_texte}\"")
    
    # Prédiction catégorie
    pred_cat = predire_ticket(exemple_texte, 'categorie')
    print(f"\n🏷️  Catégorie prédite : {pred_cat['prediction']}")
    print(f"   Confiance : {pred_cat['confiance']*100:.1f}%")
    print("   Top 3 :")
    for cat, prob in pred_cat['top_3']:
        print(f"      - {cat}: {prob*100:.1f}%")
    
    # Prédiction urgence
    pred_urg = predire_ticket(exemple_texte, 'urgence')
    print(f"\n⚡ Urgence prédite : {pred_urg['prediction']}")
    print(f"   Confiance : {pred_urg['confiance']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("=" * 70)
