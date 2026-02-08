# 📊 Rapport Complet - Projet Analyse Intelligente de Tickets Support

**Date :** 7 Février 2026  
**Équipe :** Projet Data Science - MAPHAR  
**Repository :** github.com/ahmedsamaka-ops/Analyse-intelligente-de-tickets-DS

---

## 1. 🎯 Objectif du Projet

Développer un système intelligent capable de :
1. **Classifier automatiquement** les tickets support par catégorie
2. **Prédire le niveau d'urgence** des tickets
3. **Estimer le temps de résolution** en heures

---

## 2. 📁 Données Utilisées

### Source
- **Fichier original :** `jira_backlog_v2.csv` (export JIRA)
- **Fichier nettoyé :** `data/tickets_cleaned.csv`

### Volume
| Dataset | Tickets | Pourcentage |
|---------|---------|-------------|
| Train | 536 | 70% |
| Test | 115 | 15% |
| Validation | 115 | 15% |
| **Total** | **766** | 100% |

### Variables
| Colonne | Description | Type |
|---------|-------------|------|
| `texte` | Description du ticket | Texte libre |
| `titre` | Titre du ticket | Texte libre |
| `categorie` | Catégorie du problème | 54 catégories |
| `urgence` | Niveau d'urgence | Basse/Moyenne/Très haute |
| `temps_resolution` | Temps réel de résolution | Heures (numérique) |
| `type_ticket` | Type | Demande/Incident |

---

## 3. 🔬 Méthodologie

### 3.1 Prétraitement des Données
```
1. Nettoyage du texte (minuscules, caractères spéciaux)
2. Correction des problèmes d'encodage (accents)
3. Vectorisation TF-IDF (max_features=3000, ngram_range=(1,2))
4. Encodage des labels (LabelEncoder)
```

### 3.2 Algorithmes Testés
- Naive Bayes (MultinomialNB)
- Random Forest
- Logistic Regression
- SVM (Support Vector Machine)
- Gradient Boosting
- XGBoost
- LightGBM
- Ridge Regression (pour le temps)

### 3.3 Validation
- Validation croisée 5-fold
- Évaluation sur set de validation indépendant (15%)

---

## 4. 📈 Résultats par Tâche

### 4.1 🏷️ Classification des Catégories

#### Problème Initial
- 54 catégories différentes
- Moyenne de ~14 tickets par catégorie (insuffisant)
- **Accuracy V1 : 66.09%**

#### Solution Appliquée
Regroupement sémantique des 54 catégories en **11 macro-catégories** :

| Macro-Catégorie | Catégories Regroupées |
|-----------------|----------------------|
| Gestion Comptes AD | Création compte AD, Compte AD, Compte AD désactivation |
| Accès & Partages | Accès au partage, Partage |
| Réseau & Connexion | Réseau, Connexion internet, VPN, Wifi |
| Impressions & Scanner | Toutes catégories impression |
| Applications & SAP | Applications, accès SAP, SAP |
| Matériel | Laptop, PC, Écran |
| Téléphonie | Utilitaires/Request (lignes téléphoniques) |
| Sécurité | Sécurité/Sophos, MDP |
| Projets & Dev | Création de projet |
| Messagerie | Outlook, Email |
| Autre | Reste |

#### Résultat Final
| Métrique | V1 (54 cat.) | V3 (11 cat.) | Amélioration |
|----------|--------------|--------------|--------------|
| **Accuracy** | 66.09% | **73.91%** | **+7.82%** ✅ |
| Modèle | Naive Bayes | Random Forest | - |

```
📊 Matrice de Confusion (extrait) :
                      Prédit
Réel              Gestion_AD  Accès  Réseau  ...
Gestion Comptes AD     18       2      0
Accès & Partages        1      15      1
Réseau & Connexion      0       0     12
```

---

### 4.2 🚨 Classification de l'Urgence

#### Problème Identifié : Déséquilibre des Classes
| Urgence | Nombre | Pourcentage |
|---------|--------|-------------|
| **Basse** | 726 | **94.78%** |
| Moyenne | 34 | 4.44% |
| Très haute | 6 | 0.78% |

> ⚠️ **Attention :** Avec 95% de tickets "Basse", un modèle naïf qui prédit toujours "Basse" obtient 95% d'accuracy !

#### Solution Appliquée
1. **class_weight='balanced'** : Pondération inverse des classes
2. **SMOTE** : Sur-échantillonnage synthétique des classes minoritaires
3. **Changement de métrique** : F1-Macro au lieu de l'Accuracy

#### Résultat Final
| Métrique | Avant (biaisé) | Après (équilibré) | Amélioration |
|----------|----------------|-------------------|--------------|
| Accuracy | 96.52% | 95.65% | -0.87% |
| **F1-Macro** | ~33% | **68.64%** | **+35%** ✅ |
| **Balanced Accuracy** | ~50% | **73.20%** | **+23%** ✅ |
| Recall "Moyenne" | 25% | **50%** | **+25%** ✅ |

```
📊 Comparaison des Méthodes :
Méthode                     F1-Macro    Balanced Acc
─────────────────────────────────────────────────────
Baseline (toujours Basse)     33.00%        33.00%
RF + class_weight             65.77%        62.05%
RF + SMOTE                    65.77%        62.05%
Logistic Regression 🏆        68.64%        73.20%
```

---

### 4.3 ⏱️ Prédiction du Temps de Résolution

#### Approche
- Régression avec Ridge Regression
- Features : TF-IDF du texte + type_ticket encodé

#### Résultat
| Métrique | Valeur |
|----------|--------|
| **RMSE** | **16.18 heures** |
| MAE | 10.45 heures |
| R² | 0.12 |

> Note : Le R² faible (0.12) indique que le temps de résolution dépend de facteurs non présents dans le texte (disponibilité technicien, complexité réelle, etc.)

---

## 5. 🗂️ Modèles Livrés

### Fichiers dans `/models/`

| Fichier | Description | Tâche |
|---------|-------------|-------|
| `classification_categorie_model_v3.pkl` | Random Forest (11 catégories) | Catégorie |
| `tfidf_vectorizer_categorie_v3.pkl` | Vectoriseur TF-IDF | Catégorie |
| `label_encoder_categorie_v3.pkl` | Encodeur labels | Catégorie |
| `mapping_categories.pkl` | Mapping 54→11 catégories | Catégorie |
| `classification_urgence_balanced.pkl` | Logistic Regression équilibré | Urgence |
| `tfidf_vectorizer_urgence_balanced.pkl` | Vectoriseur TF-IDF | Urgence |
| `label_encoder_urgence_balanced.pkl` | Encodeur labels | Urgence |
| `regression_temps_model.pkl` | Ridge Regression | Temps |
| `tfidf_vectorizer_regression.pkl` | Vectoriseur TF-IDF | Temps |

---

## 6. 📋 Scripts Développés

| Script | Fonction |
|--------|----------|
| `src/ml/clean_data.py` | Nettoyage et split des données |
| `src/ml/train_classifier.py` | Classification V1 (54 catégories) |
| `src/ml/train_classifier_v2.py` | Tests boosting (XGBoost, LightGBM) |
| `src/ml/train_classifier_v3.py` | Classification V3 (11 catégories) ✅ |
| `src/ml/train_regression.py` | Régression temps de résolution |
| `src/ml/train_urgence_balanced.py` | Classification urgence équilibrée ✅ |

---

## 7. 📊 Résumé des Performances Finales

| Tâche | Modèle | Métrique | Score |
|-------|--------|----------|-------|
| **Catégorie** | Random Forest | Accuracy | **73.91%** ✅ |
| **Urgence** | Logistic Regression | F1-Macro | **68.64%** ✅ |
| **Temps** | Ridge Regression | RMSE | **16.18h** |

---

## 8. 💡 Leçons Apprises

### Ce qui a fonctionné ✅
1. **Regroupement des catégories** : +7.82% d'accuracy
2. **Équilibrage des classes** : F1-Macro doublé
3. **TF-IDF avec n-grams** : Capture les expressions (ex: "compte AD")

### Ce qui n'a pas fonctionné ❌
1. **Boosting (XGBoost, LightGBM)** : Pas d'amélioration vs Random Forest
2. **Plus de features TF-IDF** : Overfitting
3. **SVM** : Trop lent, résultats similaires

### Limitations 📉
1. **Données limitées** : 766 tickets seulement
2. **Déséquilibre urgence** : 95% "Basse"
3. **Temps de résolution** : Dépend de facteurs externes

---

## 9. 🚀 Prochaines Étapes Recommandées

1. **Application Streamlit** : Interface utilisateur pour les prédictions
2. **Collecte de données** : Plus de tickets urgents
3. **Feedback loop** : Amélioration continue avec les corrections utilisateurs
4. **API REST** : Intégration avec JIRA

---

## 10. 📎 Annexes

### Structure du Projet
```
Analyse_intelligente_de_tickets_DS/
├── data/
│   ├── tickets_cleaned.csv
│   ├── train.csv (70%)
│   ├── test.csv (15%)
│   └── validation.csv (15%)
├── models/
│   └── [12 fichiers .pkl]
├── src/ml/
│   └── [6 scripts Python]
├── docs/
│   ├── Specifications_Fonctionnelles.md
│   └── RAPPORT_ANALYSE_URGENCE.md
├── requirements.txt
└── README.md
```

### Dépendances
```
pandas
numpy
scikit-learn
joblib
imbalanced-learn (SMOTE)
```

---

**Rapport généré le 7 Février 2026**  
*Projet Analyse Intelligente de Tickets Support - Data Science*
