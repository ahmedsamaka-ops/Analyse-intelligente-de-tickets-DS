# =============================================================================
# Script d'Entraînement du Modèle de Régression
# Projet : Analyse Intelligente de Tickets Support
# Auteur : Expert ML
# Date : Février 2026
# =============================================================================
# Ce script entraîne un modèle de régression pour prédire le temps de résolution
# d'un ticket support en heures.
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

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
    
    texte = texte.lower()
    texte = re.sub(r'[^\w\s\-àâäéèêëïîôùûüç]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte


def charger_donnees():
    """
    Charge les données d'entraînement, test et validation.
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
    
    return train_df, test_df, val_df


def analyser_temps_resolution(df):
    """
    Analyse statistique du temps de résolution.
    """
    print("\n📊 Statistiques du temps de résolution (heures) :")
    temps = df['temps_resolution']
    
    print(f"   - Minimum  : {temps.min():.2f}h")
    print(f"   - Maximum  : {temps.max():.2f}h")
    print(f"   - Moyenne  : {temps.mean():.2f}h")
    print(f"   - Médiane  : {temps.median():.2f}h")
    print(f"   - Écart-type: {temps.std():.2f}h")
    
    # Percentiles
    print(f"\n   Percentiles :")
    for p in [25, 50, 75, 90, 95]:
        print(f"      {p}% : {temps.quantile(p/100):.2f}h")
    
    return temps


def preparer_features(df, tfidf_vectorizer=None, label_encoders=None, fit=True):
    """
    Prépare les features pour la régression.
    
    Combine :
    - Features textuelles (TF-IDF)
    - Features catégorielles encodées (catégorie, urgence, type_ticket)
    - Features numériques (nb_mots)
    
    Returns:
        X: Matrice de features
        vectorizer: TF-IDF vectorizer
        encoders: Dictionnaire des label encoders
    """
    df = df.copy()
    
    # Nettoyer le texte
    df['texte_clean'] = df['texte'].apply(nettoyer_texte)
    
    # 1. Features TF-IDF
    if fit:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        X_tfidf = tfidf_vectorizer.fit_transform(df['texte_clean'])
    else:
        X_tfidf = tfidf_vectorizer.transform(df['texte_clean'])
    
    # 2. Features catégorielles
    if fit:
        label_encoders = {}
        
    categorical_features = []
    for col in ['categorie', 'urgence', 'type_ticket']:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].fillna('Inconnu'))
                label_encoders[col] = le
            else:
                # Gérer les valeurs inconnues
                le = label_encoders[col]
                encoded = df[col].fillna('Inconnu').apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                ).values
            categorical_features.append(encoded.reshape(-1, 1))
    
    # 3. Features numériques
    numeric_features = []
    if 'nb_mots' in df.columns:
        numeric_features.append(df['nb_mots'].fillna(0).values.reshape(-1, 1))
    
    # Combiner toutes les features
    X_tfidf_dense = X_tfidf.toarray()
    
    all_features = [X_tfidf_dense]
    all_features.extend(categorical_features)
    all_features.extend(numeric_features)
    
    X = np.hstack(all_features)
    
    return X, tfidf_vectorizer, label_encoders


# =============================================================================
# 2. ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION
# =============================================================================

def entrainer_regresseur(train_df, test_df, val_df):
    """
    Entraîne un modèle de régression pour prédire le temps de résolution.
    
    Returns:
        Dictionnaire avec le modèle, le vectoriseur, les encodeurs et les métriques
    """
    print("\n" + "=" * 70)
    print("⏱️  RÉGRESSION - PRÉDICTION DU TEMPS DE RÉSOLUTION")
    print("=" * 70)
    
    # Analyse des données
    analyser_temps_resolution(train_df)
    
    # Préparer les features
    print("\n🔄 Préparation des features...")
    X_train, tfidf_vectorizer, label_encoders = preparer_features(
        train_df, fit=True
    )
    y_train = train_df['temps_resolution'].values
    
    X_test, _, _ = preparer_features(
        test_df, tfidf_vectorizer, label_encoders, fit=False
    )
    y_test = test_df['temps_resolution'].values
    
    X_val, _, _ = preparer_features(
        val_df, tfidf_vectorizer, label_encoders, fit=False
    )
    y_val = val_df['temps_resolution'].values
    
    print(f"✅ Shape X_train : {X_train.shape}")
    print(f"✅ Shape X_test  : {X_test.shape}")
    print(f"✅ Shape X_val   : {X_val.shape}")
    
    # Comparaison de plusieurs modèles
    print("\n🔬 Comparaison des modèles de régression...")
    
    modeles = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE)
    }
    
    meilleurs_resultats = {'modele': None, 'nom': '', 'rmse': float('inf')}
    
    for nom, modele in modeles.items():
        print(f"\n   📊 {nom}...")
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"      RMSE : {rmse:.2f}h")
        print(f"      MAE  : {mae:.2f}h")
        print(f"      R²   : {r2:.4f}")
        
        if rmse < meilleurs_resultats['rmse']:
            meilleurs_resultats = {
                'modele': modele,
                'nom': nom,
                'rmse': rmse
            }
    
    # Utiliser le meilleur modèle
    meilleur_modele = meilleurs_resultats['modele']
    print(f"\n🏆 Meilleur modèle : {meilleurs_resultats['nom']}")
    
    # Évaluation finale sur le set de validation
    print("\n" + "-" * 50)
    print("📈 ÉVALUATION FINALE (Set de Validation)")
    print("-" * 50)
    
    y_val_pred = meilleur_modele.predict(X_val)
    
    # Appliquer les contraintes métier
    # Temps minimum : 0.5h, Temps maximum : 168h (1 semaine)
    y_val_pred = np.clip(y_val_pred, 0.5, 168)
    
    # Métriques
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    # MAPE (Mean Absolute Percentage Error) - éviter division par zéro
    mask = y_val > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_val[mask] - y_val_pred[mask]) / y_val[mask])) * 100
    else:
        mape = 0
    
    print(f"\n   📊 RMSE  : {rmse:.2f} heures")
    print(f"   📊 MAE   : {mae:.2f} heures")
    print(f"   📊 R²    : {r2:.4f} ({r2*100:.2f}%)")
    print(f"   📊 MAPE  : {mape:.2f}%")
    
    # Analyse des erreurs
    erreurs = np.abs(y_val - y_val_pred)
    print(f"\n   📊 Analyse des erreurs absolues :")
    print(f"      - Erreur moyenne     : {erreurs.mean():.2f}h")
    print(f"      - Erreur médiane     : {np.median(erreurs):.2f}h")
    print(f"      - 90% des erreurs <  : {np.percentile(erreurs, 90):.2f}h")
    
    # Pourcentage de prédictions dans une marge acceptable
    for marge in [2, 5, 10, 24]:
        pct = (erreurs <= marge).mean() * 100
        print(f"      - Erreur <= {marge:2d}h     : {pct:.1f}%")
    
    # Cross-validation
    print("\n   🔄 Cross-Validation (5-fold) sur Train...")
    cv_scores = cross_val_score(
        meilleur_modele, X_train, y_train, 
        cv=5, scoring='neg_root_mean_squared_error'
    )
    cv_rmse = -cv_scores
    print(f"      RMSE CV : {cv_rmse}")
    print(f"      Moyenne : {cv_rmse.mean():.2f}h (+/- {cv_rmse.std()*2:.2f}h)")
    
    # Feature importance (si Random Forest)
    if hasattr(meilleur_modele, 'feature_importances_'):
        print("\n   🔍 Top 10 Features les plus importantes :")
        importances = meilleur_modele.feature_importances_
        
        # Créer les noms des features
        feature_names = list(tfidf_vectorizer.get_feature_names_out())
        feature_names.extend(['categorie_encoded', 'urgence_encoded', 'type_ticket_encoded', 'nb_mots'])
        
        # Trier par importance
        indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(indices):
            if idx < len(feature_names):
                print(f"      {i+1:2d}. {feature_names[idx][:30]:<30} : {importances[idx]:.4f}")
    
    # Sauvegarder les modèles
    print("\n💾 Sauvegarde des modèles...")
    
    modele_path = os.path.join(MODELS_DIR, "regression_temps_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer_regression.pkl")
    encoders_path = os.path.join(MODELS_DIR, "label_encoders_regression.pkl")
    
    joblib.dump(meilleur_modele, modele_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    joblib.dump(label_encoders, encoders_path)
    
    print(f"   ✅ Modèle     : {modele_path}")
    print(f"   ✅ Vectorizer : {vectorizer_path}")
    print(f"   ✅ Encoders   : {encoders_path}")
    
    return {
        'modele': meilleur_modele,
        'vectorizer': tfidf_vectorizer,
        'encoders': label_encoders,
        'metriques': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        },
        'nom_modele': meilleurs_resultats['nom']
    }


# =============================================================================
# 3. FONCTION DE PRÉDICTION
# =============================================================================

def predire_temps_resolution(texte, categorie='Autre', urgence='Basse', type_ticket='Demande', nb_mots=None):
    """
    Prédit le temps de résolution d'un ticket.
    
    Args:
        texte: Le texte du ticket
        categorie: La catégorie du ticket
        urgence: Le niveau d'urgence
        type_ticket: Type (Demande ou Incident)
        nb_mots: Nombre de mots (calculé si None)
        
    Returns:
        Dictionnaire avec la prédiction et l'intervalle de confiance
    """
    # Charger les modèles
    modele = joblib.load(os.path.join(MODELS_DIR, "regression_temps_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_regression.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders_regression.pkl"))
    
    # Préparer le texte
    texte_clean = nettoyer_texte(texte)
    
    # Calculer nb_mots si non fourni
    if nb_mots is None:
        nb_mots = len(texte_clean.split())
    
    # Créer un DataFrame temporaire
    temp_df = pd.DataFrame({
        'texte': [texte],
        'categorie': [categorie],
        'urgence': [urgence],
        'type_ticket': [type_ticket],
        'nb_mots': [nb_mots]
    })
    
    # Préparer les features
    X, _, _ = preparer_features(temp_df, vectorizer, encoders, fit=False)
    
    # Prédiction
    prediction = modele.predict(X)[0]
    
    # Appliquer les contraintes métier
    prediction = np.clip(prediction, 0.5, 168)
    
    # Calculer l'intervalle de confiance (±20% par défaut)
    marge = 0.20
    intervalle_min = max(0.5, prediction * (1 - marge))
    intervalle_max = min(168, prediction * (1 + marge))
    
    return {
        'temps_estime_heures': float(prediction),
        'temps_estime_jours': float(prediction / 24),
        'intervalle_confiance': {
            'min_heures': float(intervalle_min),
            'max_heures': float(intervalle_max),
            'min_jours': float(intervalle_min / 24),
            'max_jours': float(intervalle_max / 24)
        }
    }


# =============================================================================
# 4. MAIN - EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION")
    print("   Projet : Analyse Intelligente de Tickets Support")
    print("   Date   :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Charger les données
    train_df, test_df, val_df = charger_donnees()
    
    # Entraîner le régresseur
    resultats = entrainer_regresseur(train_df, test_df, val_df)
    
    # Résumé final
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ FINAL - MODÈLE DE RÉGRESSION")
    print("=" * 70)
    
    print(f"\n⏱️  Régression TEMPS DE RÉSOLUTION :")
    print(f"   - Modèle utilisé : {resultats['nom_modele']}")
    print(f"   - RMSE           : {resultats['metriques']['rmse']:.2f} heures")
    print(f"   - MAE            : {resultats['metriques']['mae']:.2f} heures")
    print(f"   - R²             : {resultats['metriques']['r2']*100:.2f}%")
    print(f"   - MAPE           : {resultats['metriques']['mape']:.2f}%")
    print(f"   - CV RMSE        : {resultats['metriques']['cv_rmse_mean']:.2f}h (+/- {resultats['metriques']['cv_rmse_std']*2:.2f}h)")
    
    print("\n📁 Fichiers sauvegardés dans le dossier 'models/':")
    for f in os.listdir(MODELS_DIR):
        if 'regression' in f.lower() and f.endswith('.pkl'):
            size = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
            print(f"   - {f} ({size:.1f} KB)")
    
    # Test avec des exemples
    print("\n" + "=" * 70)
    print("🧪 TESTS AVEC DES EXEMPLES")
    print("=" * 70)
    
    exemples = [
        {
            'texte': "problème de connexion wifi impossible de se connecter",
            'categorie': 'Connexion internet',
            'urgence': 'Moyenne',
            'type_ticket': 'Incident'
        },
        {
            'texte': "demande de création d'un compte active directory pour nouveau collaborateur",
            'categorie': 'Création compte AD',
            'urgence': 'Basse',
            'type_ticket': 'Demande'
        },
        {
            'texte': "panne totale réseau urgent toute l'équipe bloquée",
            'categorie': 'Connexion internet',
            'urgence': 'Très haute',
            'type_ticket': 'Incident'
        }
    ]
    
    for i, ex in enumerate(exemples, 1):
        print(f"\n📝 Exemple {i} : \"{ex['texte'][:50]}...\"")
        print(f"   Catégorie : {ex['categorie']}, Urgence : {ex['urgence']}")
        
        pred = predire_temps_resolution(
            ex['texte'], 
            ex['categorie'], 
            ex['urgence'], 
            ex['type_ticket']
        )
        
        print(f"   ⏱️  Temps estimé : {pred['temps_estime_heures']:.1f}h ({pred['temps_estime_jours']:.1f} jours)")
        print(f"   📊 Intervalle   : [{pred['intervalle_confiance']['min_heures']:.1f}h - {pred['intervalle_confiance']['max_heures']:.1f}h]")
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("=" * 70)
