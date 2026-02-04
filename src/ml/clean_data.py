# =============================================================================
# Script de Nettoyage et Préparation des Données GLPI
# Projet : Analyse Intelligente de Tickets Support
# Auteur : Personne A (Expert ML)
# Date : Février 2026
# =============================================================================

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("=" * 60)
print("ÉTAPE 1 : Chargement des données")
print("=" * 60)

# Charger le fichier CSV avec le bon séparateur et encodage
df = pd.read_csv(
    'data/tickets_glpi.csv', 
    sep=';', 
    encoding='utf-8',
    on_bad_lines='skip'
)

print(f"✅ Fichier chargé avec succès !")
print(f"   - Nombre de tickets : {len(df)}")
print(f"   - Nombre de colonnes : {len(df.columns)}")
print(f"\n📋 Colonnes disponibles :")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# =============================================================================
# 2. SÉLECTION DES COLONNES UTILES
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 2 : Sélection des colonnes utiles")
print("=" * 60)

# Colonnes à conserver pour le projet
colonnes_utiles = [
    'ID',
    'Titre',
    'Description',
    'Catégorie',
    'Urgence',
    'Priorité',
    'Impact',
    'Type',
    'Statut',
    'Temps_resolution_hrs'
]

# Vérifier quelles colonnes existent
colonnes_existantes = [col for col in colonnes_utiles if col in df.columns]
colonnes_manquantes = [col for col in colonnes_utiles if col not in df.columns]

print(f"✅ Colonnes trouvées : {len(colonnes_existantes)}")
if colonnes_manquantes:
    print(f"⚠️  Colonnes manquantes : {colonnes_manquantes}")

# Créer le DataFrame nettoyé
df_clean = df[colonnes_existantes].copy()

# =============================================================================
# 3. NETTOYAGE DU TEXTE
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 3 : Nettoyage du texte")
print("=" * 60)

def nettoyer_texte(texte):
    """Nettoie un texte pour l'analyse ML"""
    if pd.isna(texte):
        return ""
    
    # Convertir en string
    texte = str(texte)
    
    # Corriger les problèmes d'encodage courants
    corrections = {
        'Ã©': 'é', 'Ã¨': 'è', 'Ãª': 'ê', 'Ã ': 'à',
        'Ã§': 'ç', 'Ã´': 'ô', 'Ã®': 'î', 'Ã¯': 'ï',
        'Ã¹': 'ù', 'Ã»': 'û', 'Ã¢': 'â', 'â€™': "'",
        'Â': '', 'â€"': '-', 'â€œ': '"', 'â€': '"'
    }
    for ancien, nouveau in corrections.items():
        texte = texte.replace(ancien, nouveau)
    
    # Mettre en minuscules
    texte = texte.lower()
    
    # Supprimer les caractères spéciaux inutiles (garder lettres, chiffres, espaces)
    texte = re.sub(r'[^\w\s\-\'àâäéèêëïîôùûüç]', ' ', texte)
    
    # Supprimer les espaces multiples
    texte = re.sub(r'\s+', ' ', texte)
    
    # Supprimer les espaces en début/fin
    texte = texte.strip()
    
    return texte

# Appliquer le nettoyage
df_clean['Titre_clean'] = df_clean['Titre'].apply(nettoyer_texte)

# Nettoyer la description si elle existe
if 'Description' in df_clean.columns:
    df_clean['Description_clean'] = df_clean['Description'].apply(nettoyer_texte)
    # Combiner Titre + Description pour avoir plus de contexte
    df_clean['Texte_complet'] = df_clean['Titre_clean'] + ' ' + df_clean['Description_clean']
else:
    df_clean['Texte_complet'] = df_clean['Titre_clean']

# Nettoyer le texte complet
df_clean['Texte_complet'] = df_clean['Texte_complet'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

print(f"✅ Texte nettoyé pour {len(df_clean)} tickets")

# Exemple de nettoyage
print("\n📝 Exemple de nettoyage :")
print(f"   AVANT : {df_clean['Titre'].iloc[0][:80]}...")
print(f"   APRÈS : {df_clean['Titre_clean'].iloc[0][:80]}...")

# =============================================================================
# 4. NETTOYAGE DES CATÉGORIES
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 4 : Nettoyage des catégories")
print("=" * 60)

# Nettoyer la colonne Catégorie
if 'Catégorie' in df_clean.columns:
    # Remplacer les valeurs vides
    df_clean['Catégorie'] = df_clean['Catégorie'].fillna('Autre')
    df_clean['Catégorie'] = df_clean['Catégorie'].replace('', 'Autre')
    
    # Simplifier les catégories longues (prendre le premier niveau)
    def simplifier_categorie(cat):
        if pd.isna(cat) or cat == '':
            return 'Autre'
        # Prendre le premier niveau si c'est une hiérarchie (ex: "A > B > C" -> "A")
        if '>' in str(cat):
            return str(cat).split('>')[0].strip()
        return str(cat).strip()
    
    df_clean['Categorie_simple'] = df_clean['Catégorie'].apply(simplifier_categorie)
    
    # Afficher la distribution
    print("\n📊 Distribution des catégories simplifiées :")
    distribution = df_clean['Categorie_simple'].value_counts()
    for cat, count in distribution.head(15).items():
        pct = count / len(df_clean) * 100
        print(f"   {cat[:40]:<40} : {count:4d} ({pct:5.1f}%)")
    
    if len(distribution) > 15:
        print(f"   ... et {len(distribution) - 15} autres catégories")

# =============================================================================
# 5. NETTOYAGE DU TEMPS DE RÉSOLUTION
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 5 : Nettoyage du temps de résolution")
print("=" * 60)

if 'Temps_resolution_hrs' in df_clean.columns:
    # Convertir en numérique (gérer les virgules françaises)
    df_clean['Temps_resolution_hrs'] = df_clean['Temps_resolution_hrs'].astype(str)
    df_clean['Temps_resolution_hrs'] = df_clean['Temps_resolution_hrs'].str.replace(',', '.')
    df_clean['Temps_resolution_hrs'] = pd.to_numeric(df_clean['Temps_resolution_hrs'], errors='coerce')
    
    # Statistiques
    temps_valide = df_clean['Temps_resolution_hrs'].dropna()
    print(f"✅ Temps de résolution analysé :")
    print(f"   - Valeurs valides : {len(temps_valide)} / {len(df_clean)}")
    print(f"   - Minimum : {temps_valide.min():.2f} heures")
    print(f"   - Maximum : {temps_valide.max():.2f} heures")
    print(f"   - Moyenne : {temps_valide.mean():.2f} heures")
    print(f"   - Médiane : {temps_valide.median():.2f} heures")
    
    # Remplacer les valeurs 0 ou négatives par la médiane (optionnel)
    mediane = temps_valide.median()
    df_clean.loc[df_clean['Temps_resolution_hrs'] <= 0, 'Temps_resolution_hrs'] = mediane

# =============================================================================
# 6. NETTOYAGE DE L'URGENCE ET PRIORITÉ
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 6 : Nettoyage Urgence et Priorité")
print("=" * 60)

if 'Urgence' in df_clean.columns:
    df_clean['Urgence'] = df_clean['Urgence'].fillna('Basse')
    df_clean['Urgence'] = df_clean['Urgence'].str.strip()
    
    # Corriger les problèmes d'encodage
    df_clean['Urgence'] = df_clean['Urgence'].replace({
        'TrÃ¨s haute': 'Très haute',
        'TrÃ©s haute': 'Très haute'
    })
    
    print("\n📊 Distribution de l'Urgence :")
    for urgence, count in df_clean['Urgence'].value_counts().items():
        pct = count / len(df_clean) * 100
        print(f"   {urgence:<15} : {count:4d} ({pct:5.1f}%)")

# =============================================================================
# 7. SUPPRESSION DES TICKETS TROP COURTS
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 7 : Filtrage des tickets")
print("=" * 60)

# Compter les mots dans le texte
df_clean['nb_mots'] = df_clean['Texte_complet'].apply(lambda x: len(str(x).split()))

# Statistiques
print(f"📊 Statistiques sur la longueur des textes :")
print(f"   - Minimum : {df_clean['nb_mots'].min()} mots")
print(f"   - Maximum : {df_clean['nb_mots'].max()} mots")
print(f"   - Moyenne : {df_clean['nb_mots'].mean():.1f} mots")

# Filtrer les tickets trop courts (moins de 3 mots)
tickets_courts = len(df_clean[df_clean['nb_mots'] < 3])
print(f"\n⚠️  Tickets avec moins de 3 mots : {tickets_courts}")

# On garde tous les tickets pour l'instant (même courts)
# df_clean = df_clean[df_clean['nb_mots'] >= 3]

# =============================================================================
# 8. CRÉATION DU DATASET FINAL
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 8 : Création du dataset final")
print("=" * 60)

# Colonnes finales pour le ML
colonnes_finales = [
    'ID',
    'Texte_complet',
    'Titre_clean',
    'Categorie_simple' if 'Categorie_simple' in df_clean.columns else 'Catégorie',
    'Urgence',
    'Temps_resolution_hrs',
    'Type',
    'nb_mots'
]

# Garder seulement les colonnes qui existent
colonnes_finales = [c for c in colonnes_finales if c in df_clean.columns]
df_final = df_clean[colonnes_finales].copy()

# Renommer pour clarté
df_final = df_final.rename(columns={
    'Texte_complet': 'texte',
    'Titre_clean': 'titre',
    'Categorie_simple': 'categorie',
    'Urgence': 'urgence',
    'Temps_resolution_hrs': 'temps_resolution',
    'Type': 'type_ticket'
})

print(f"✅ Dataset final créé avec {len(df_final)} tickets")
print(f"   Colonnes : {list(df_final.columns)}")

# =============================================================================
# 9. SPLIT TRAIN / VALIDATION / TEST
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 9 : Division Train / Validation / Test")
print("=" * 60)

# 70% train, 15% validation, 15% test
train_val, test = train_test_split(df_final, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.176, random_state=42)  # 0.176 de 85% ≈ 15%

print(f"✅ Division effectuée :")
print(f"   - Train      : {len(train):4d} tickets ({len(train)/len(df_final)*100:.1f}%)")
print(f"   - Validation : {len(val):4d} tickets ({len(val)/len(df_final)*100:.1f}%)")
print(f"   - Test       : {len(test):4d} tickets ({len(test)/len(df_final)*100:.1f}%)")

# =============================================================================
# 10. SAUVEGARDE DES FICHIERS
# =============================================================================
print("\n" + "=" * 60)
print("ÉTAPE 10 : Sauvegarde des fichiers")
print("=" * 60)

# Sauvegarder
df_final.to_csv('data/tickets_cleaned.csv', index=False, encoding='utf-8')
train.to_csv('data/train.csv', index=False, encoding='utf-8')
val.to_csv('data/validation.csv', index=False, encoding='utf-8')
test.to_csv('data/test.csv', index=False, encoding='utf-8')

print(f"✅ Fichiers sauvegardés dans le dossier 'data/' :")
print(f"   - tickets_cleaned.csv (dataset complet nettoyé)")
print(f"   - train.csv ({len(train)} tickets)")
print(f"   - validation.csv ({len(val)} tickets)")
print(f"   - test.csv ({len(test)} tickets)")

# =============================================================================
# 11. RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 60)
print("✅ NETTOYAGE TERMINÉ AVEC SUCCÈS !")
print("=" * 60)
print(f"""
📊 RÉSUMÉ DU DATASET NETTOYÉ :
   - Total tickets : {len(df_final)}
   - Colonnes : texte, titre, categorie, urgence, temps_resolution, type_ticket
   - Catégories uniques : {df_final['categorie'].nunique() if 'categorie' in df_final.columns else 'N/A'}
   - Temps moyen résolution : {df_final['temps_resolution'].mean():.2f}h

🎯 PROCHAINES ÉTAPES :
   1. Exécuter src/ml/train_model.py pour entraîner le classificateur
   2. Exécuter src/ml/regression.py pour le modèle de régression
   3. Exécuter src/rag/ingest.py pour créer la base vectorielle
""")
