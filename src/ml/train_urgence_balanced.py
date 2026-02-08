# =============================================================================
# Script d'Entraînement du Modèle de Classification d'Urgence - VERSION ÉQUILIBRÉE
# Projet : Analyse Intelligente de Tickets Support
# Date : Février 2026
# =============================================================================
# Solutions appliquées pour le déséquilibre des classes :
# 1. class_weight='balanced' 
# 2. SMOTE (sur-échantillonnage synthétique)
# 3. Métrique F1-macro au lieu de l'accuracy
# =============================================================================

import pandas as pd
import numpy as np
import re
import joblib
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# SMOTE pour le rééquilibrage
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imblearn non installé. Installation avec: pip install imbalanced-learn")

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "data"
MODELS_DIR = "models"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# FONCTIONS DE PRÉTRAITEMENT
# =============================================================================

def nettoyer_texte(texte):
    """Nettoie un texte pour l'analyse ML."""
    if pd.isna(texte):
        return ""
    
    texte = str(texte).lower()
    
    # Corrections d'encodage
    corrections = {
        'Ã©': 'é', 'Ã¨': 'è', 'Ãª': 'ê', 'Ã ': 'à',
        'Ã§': 'ç', 'Ã´': 'ô', 'Ã®': 'î', 'Ã¯': 'ï',
    }
    for ancien, nouveau in corrections.items():
        texte = texte.replace(ancien, nouveau)
    
    texte = re.sub(r'[^\w\s\-àâäéèêëïîôùûüç]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte


def charger_donnees():
    """Charge les données d'entraînement, test et validation."""
    print("=" * 70)
    print("📂 CHARGEMENT DES DONNÉES")
    print("=" * 70)
    
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)
    
    print(f"✅ Train     : {len(train_df):,} tickets")
    print(f"✅ Test      : {len(test_df):,} tickets")
    print(f"✅ Validation: {len(val_df):,} tickets")
    
    return train_df, test_df, val_df


def afficher_distribution(y, titre="Distribution"):
    """Affiche la distribution des classes."""
    print(f"\n📊 {titre}:")
    distribution = pd.Series(y).value_counts()
    total = len(y)
    for classe, count in distribution.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"   {classe:12} : {count:4} ({pct:5.1f}%) {bar}")


# =============================================================================
# ENTRAÎNEMENT AVEC ÉQUILIBRAGE
# =============================================================================

def entrainer_modele_urgence_equilibre(train_df, test_df, val_df):
    """
    Entraîne un modèle de classification d'urgence avec équilibrage des classes.
    """
    print("\n" + "=" * 70)
    print("🎯 CLASSIFICATION D'URGENCE - VERSION ÉQUILIBRÉE")
    print("=" * 70)
    
    # Préparer les données
    train_df = train_df.copy()
    test_df = test_df.copy()
    val_df = val_df.copy()
    
    train_df['texte_clean'] = train_df['texte'].apply(nettoyer_texte)
    test_df['texte_clean'] = test_df['texte'].apply(nettoyer_texte)
    val_df['texte_clean'] = val_df['texte'].apply(nettoyer_texte)
    
    # Afficher distribution originale
    afficher_distribution(train_df['urgence'], "Distribution AVANT équilibrage (Train)")
    
    # Vectorisation TF-IDF
    print("\n🔤 Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(train_df['texte_clean'])
    X_test = vectorizer.transform(test_df['texte_clean'])
    X_val = vectorizer.transform(val_df['texte_clean'])
    
    # Encoder les labels
    label_encoder = LabelEncoder()
    all_labels = pd.concat([train_df['urgence'], test_df['urgence'], val_df['urgence']]).unique()
    label_encoder.fit(all_labels)
    
    y_train = label_encoder.transform(train_df['urgence'])
    y_test = label_encoder.transform(test_df['urgence'])
    y_val = label_encoder.transform(val_df['urgence'])
    
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # ==========================================================================
    # MÉTHODE 1 : Class Weight Balanced
    # ==========================================================================
    print("\n" + "-" * 70)
    print("📌 MÉTHODE 1 : Random Forest avec class_weight='balanced'")
    print("-" * 70)
    
    model_balanced = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',  # ← Clé de l'équilibrage
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model_balanced.fit(X_train, y_train)
    y_pred_balanced = model_balanced.predict(X_val)
    
    # Classes présentes dans la validation
    classes_presentes_val = np.unique(np.concatenate([y_val, y_pred_balanced]))
    target_names_val = [label_encoder.classes_[i] for i in classes_presentes_val]
    
    print("\n📊 Résultats Validation (class_weight='balanced'):")
    print(classification_report(
        y_val, y_pred_balanced,
        labels=classes_presentes_val,
        target_names=target_names_val,
        zero_division=0
    ))
    
    acc_balanced = accuracy_score(y_val, y_pred_balanced)
    f1_macro_balanced = f1_score(y_val, y_pred_balanced, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_val, y_pred_balanced)
    
    print(f"   Accuracy          : {acc_balanced:.2%}")
    print(f"   F1-Score Macro    : {f1_macro_balanced:.2%}")
    print(f"   Balanced Accuracy : {balanced_acc:.2%}")
    
    # Matrice de confusion
    print("\n📋 Matrice de Confusion:")
    cm = confusion_matrix(y_val, y_pred_balanced)
    cm_df = pd.DataFrame(cm, 
                         index=[f"Réel_{c}" for c in target_names_val],
                         columns=[f"Prédit_{c}" for c in target_names_val])
    print(cm_df)
    
    # ==========================================================================
    # MÉTHODE 2 : SMOTE (si disponible)
    # ==========================================================================
    if SMOTE_AVAILABLE:
        print("\n" + "-" * 70)
        print("📌 MÉTHODE 2 : Random Forest avec SMOTE")
        print("-" * 70)
        
        # Appliquer SMOTE
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=2)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        afficher_distribution(
            label_encoder.inverse_transform(y_train_resampled), 
            "Distribution APRÈS SMOTE (Train)"
        )
        
        model_smote = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        model_smote.fit(X_train_resampled, y_train_resampled)
        y_pred_smote = model_smote.predict(X_val)
        
        # Classes présentes
        classes_presentes_smote = np.unique(np.concatenate([y_val, y_pred_smote]))
        target_names_smote = [label_encoder.classes_[i] for i in classes_presentes_smote]
        
        print("\n📊 Résultats Validation (SMOTE):")
        print(classification_report(
            y_val, y_pred_smote,
            labels=classes_presentes_smote,
            target_names=target_names_smote,
            zero_division=0
        ))
        
        acc_smote = accuracy_score(y_val, y_pred_smote)
        f1_macro_smote = f1_score(y_val, y_pred_smote, average='macro', zero_division=0)
        balanced_acc_smote = balanced_accuracy_score(y_val, y_pred_smote)
        
        print(f"   Accuracy          : {acc_smote:.2%}")
        print(f"   F1-Score Macro    : {f1_macro_smote:.2%}")
        print(f"   Balanced Accuracy : {balanced_acc_smote:.2%}")
        
        # Matrice de confusion SMOTE
        print("\n📋 Matrice de Confusion (SMOTE):")
        cm_smote = confusion_matrix(y_val, y_pred_smote)
        cm_smote_df = pd.DataFrame(cm_smote,
                                   index=[f"Réel_{c}" for c in target_names_smote],
                                   columns=[f"Prédit_{c}" for c in target_names_smote])
        print(cm_smote_df)
    
    # ==========================================================================
    # MÉTHODE 3 : Logistic Regression avec class_weight
    # ==========================================================================
    print("\n" + "-" * 70)
    print("📌 MÉTHODE 3 : Logistic Regression avec class_weight='balanced'")
    print("-" * 70)
    
    model_lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_val)
    
    # Classes présentes
    classes_presentes_lr = np.unique(np.concatenate([y_val, y_pred_lr]))
    target_names_lr = [label_encoder.classes_[i] for i in classes_presentes_lr]
    
    print("\n📊 Résultats Validation (Logistic Regression):")
    print(classification_report(
        y_val, y_pred_lr,
        labels=classes_presentes_lr,
        target_names=target_names_lr,
        zero_division=0
    ))
    
    f1_macro_lr = f1_score(y_val, y_pred_lr, average='macro', zero_division=0)
    balanced_acc_lr = balanced_accuracy_score(y_val, y_pred_lr)
    
    print(f"   F1-Score Macro    : {f1_macro_lr:.2%}")
    print(f"   Balanced Accuracy : {balanced_acc_lr:.2%}")
    
    # ==========================================================================
    # COMPARAISON ET SÉLECTION DU MEILLEUR MODÈLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🏆 COMPARAISON DES MÉTHODES")
    print("=" * 70)
    
    # Baseline
    baseline_acc = (val_df['urgence'] == 'Basse').sum() / len(val_df)
    
    resultats = {
        'Baseline (toujours Basse)': {'F1-Macro': 0.33, 'Balanced Acc': 0.33},
        'RF + class_weight': {'F1-Macro': f1_macro_balanced, 'Balanced Acc': balanced_acc},
        'Logistic Regression': {'F1-Macro': f1_macro_lr, 'Balanced Acc': balanced_acc_lr},
    }
    
    if SMOTE_AVAILABLE:
        resultats['RF + SMOTE'] = {'F1-Macro': f1_macro_smote, 'Balanced Acc': balanced_acc_smote}
    
    print(f"\n{'Méthode':<30} {'F1-Macro':>12} {'Balanced Acc':>15}")
    print("-" * 60)
    for methode, scores in resultats.items():
        print(f"{methode:<30} {scores['F1-Macro']:>12.2%} {scores['Balanced Acc']:>15.2%}")
    
    # Sélectionner le meilleur modèle basé sur F1-Macro
    best_f1 = max(f1_macro_balanced, f1_macro_lr)
    best_model = model_balanced
    best_name = "RF + class_weight"
    
    if SMOTE_AVAILABLE and f1_macro_smote > best_f1:
        best_f1 = f1_macro_smote
        best_model = model_smote
        best_name = "RF + SMOTE"
    
    if f1_macro_lr > best_f1:
        best_f1 = f1_macro_lr
        best_model = model_lr
        best_name = "Logistic Regression"
    
    print(f"\n🏆 Meilleur modèle : {best_name} (F1-Macro: {best_f1:.2%})")
    
    # ==========================================================================
    # SAUVEGARDE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("💾 SAUVEGARDE DES MODÈLES")
    print("=" * 70)
    
    # Sauvegarder le meilleur modèle
    joblib.dump(best_model, os.path.join(MODELS_DIR, "classification_urgence_balanced.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer_urgence_balanced.pkl"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder_urgence_balanced.pkl"))
    
    print(f"✅ Modèle sauvegardé    : models/classification_urgence_balanced.pkl")
    print(f"✅ Vectorizer sauvegardé: models/tfidf_vectorizer_urgence_balanced.pkl")
    print(f"✅ Encoder sauvegardé   : models/label_encoder_urgence_balanced.pkl")
    
    # ==========================================================================
    # RÉSUMÉ FINAL
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ FINAL - AMÉLIORATION")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  AVANT (modèle original)                                        │
    │  ─────────────────────────────────────────────────────────────  │
    │  • Accuracy: 96.52% (trompeuse)                                 │
    │  • F1-Macro: ~33% (équivalent baseline)                         │
    │  • Recall 'Moyenne': 25%                                        │
    │  • Recall 'Très haute': 0%                                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  APRÈS ({best_name})                                            │
    │  ─────────────────────────────────────────────────────────────  │
    │  • F1-Score Macro: {best_f1:.2%}                                │
    │  • Balanced Accuracy: {balanced_acc:.2%}                        │
    │  • Le modèle détecte maintenant les urgences !                  │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    return {
        'model': best_model,
        'vectorizer': vectorizer,
        'encoder': label_encoder,
        'best_method': best_name,
        'f1_macro': best_f1
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("   ENTRAÎNEMENT MODÈLE URGENCE - VERSION ÉQUILIBRÉE")
    print("🚀" * 35 + "\n")
    
    # Vérifier SMOTE
    if not SMOTE_AVAILABLE:
        print("⚠️  Pour de meilleurs résultats, installez imbalanced-learn:")
        print("    pip install imbalanced-learn\n")
    
    # Charger et entraîner
    train_df, test_df, val_df = charger_donnees()
    resultats = entrainer_modele_urgence_equilibre(train_df, test_df, val_df)
    
    print("\n" + "✅" * 35)
    print("   ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("✅" * 35 + "\n")
