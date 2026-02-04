# Document de Spécifications Fonctionnelles

## Plateforme d'Analyse Intelligente de Tickets Support

**Version:** 1.0  

**Date:** 31 Janvier 2026  
**Projet:** Système Hybride d'Analyse de Tickets avec ML Classique et RAG  
**Localisation:** Maroc  

---

## 1. INTRODUCTION

### 1.1 Objectif du Document

Ce document définit les spécifications fonctionnelles et non-fonctionnelles de la plateforme d'analyse intelligente de tickets support. Il sert de référence pour le développement, les tests et la validation du système.

### 1.2 Portée du Projet

Développer une plateforme web permettant l'analyse automatisée de tickets support à travers le Maroc, combinant :
- **Classification automatique** des tickets par catégorie et priorité (ML Classique)
- **Prédiction du temps de résolution** estimé (Régression)
- **Assistant conversationnel intelligent** (RAG/LLM) pour répondre aux questions sur les tickets

### 1.3 Contexte

La plateforme sera développée avec une architecture hybride combinant :
- **Machine Learning Classique** : Random Forest, XGBoost pour la classification et régression
- **Intelligence Artificielle Générative** : RAG (Retrieval-Augmented Generation) avec LLM pour le chatbot
- **Interface Web** : Streamlit pour l'expérience utilisateur

### 1.4 Public Cible

- Entreprises marocaines de télécommunications (Maroc Telecom, Inwi, Orange Maroc)
- Services client des grandes entreprises (ONCF, OCP, banques)
- Centres d'appels et support technique

---

## 2. IDENTIFICATION DES ACTEURS

### 2.1 Acteur Principal : Agent Support / Utilisateur

**Description:** Personne travaillant au service client qui soumet et analyse des tickets support.

**Responsabilités:**
- Soumettre un nouveau ticket pour analyse
- Consulter la classification automatique du ticket
- Visualiser le temps de résolution estimé
- Poser des questions au chatbot sur le contenu des tickets
- Consulter l'historique des analyses effectuées
- Recevoir des recommandations de résolution

**Caractéristiques:**
- Peut travailler en français, arabe ou darija
- Doit s'authentifier pour accéder à l'historique
- Peut utiliser le chatbot sans authentification

### 2.2 Acteur Secondaire : Superviseur / Manager

**Description:** Responsable d'équipe supervisant les agents support et les performances du système.

**Responsabilités:**
- Consulter les statistiques globales des tickets
- Analyser les tendances par catégorie
- Évaluer les performances des modèles de prédiction
- Exporter les rapports d'analyse
- Configurer les seuils d'alerte
- Gérer les utilisateurs de son équipe

**Caractéristiques:**
- Possède un rôle spécifique "SUPERVISOR"
- Accès à un tableau de bord analytique
- Peut consulter les métriques de performance des modèles

### 2.3 Acteur Tertiaire : Administrateur Système

**Description:** Responsable technique de la plateforme.

**Responsabilités:**
- Gérer tous les utilisateurs (Agents et Superviseurs)
- Mettre à jour les modèles ML
- Réentraîner les modèles avec de nouvelles données
- Gérer la base vectorielle (ChromaDB)
- Configurer le LLM et les paramètres RAG
- Monitorer les performances système

**Caractéristiques:**
- Droits d'accès complets
- Rôle "ADMIN"
- Accès aux logs et métriques techniques

### 2.4 Acteur Système : Système Automatisé

**Description:** Processus automatisés gérant les tâches de fond.

**Responsabilités:**
- Prétraiter les textes des tickets (nettoyage, tokenisation)
- Vectoriser les documents pour la base ChromaDB
- Mettre à jour les embeddings périodiquement
- Recalculer les métriques de performance
- Sauvegarder les modèles entraînés

---

## 3. EXIGENCES FONCTIONNELLES PAR ACTEUR

### 3.1 Exigences Fonctionnelles - AGENT SUPPORT

#### RF-A-001: Soumission de Ticket pour Analyse

**Priorité:** HAUTE  
**Description:** Un agent doit pouvoir soumettre un ticket pour obtenir une analyse automatique.

**Critères d'acceptation:**
- Saisie du texte du ticket via formulaire ou upload de fichier (TXT, CSV)
- Support du français, arabe et darija marocain
- Prétraitement automatique du texte (nettoyage, normalisation)
- Temps de traitement < 3 secondes
- Affichage immédiat des résultats

#### RF-A-002: Classification Automatique

**Priorité:** HAUTE  
**Description:** Le système doit classifier automatiquement le ticket.

**Critères d'acceptation:**
- Prédiction de la catégorie principale (Technique, Facturation, Commercial, RH, Réseau)
- Prédiction du niveau de priorité (Urgente, Haute, Moyenne, Basse)
- Affichage du score de confiance (pourcentage)
- Affichage des probabilités pour chaque catégorie
- Explication des mots-clés ayant influencé la décision

#### RF-A-003: Prédiction du Temps de Résolution

**Priorité:** HAUTE  
**Description:** Estimer le temps nécessaire pour résoudre le ticket.

**Critères d'acceptation:**
- Prédiction en heures ou jours
- Intervalle de confiance (min-max estimé)
- Facteurs influençant l'estimation affichés
- Comparaison avec la moyenne historique de la catégorie
- Mise à jour dynamique si le texte est modifié

#### RF-A-004: Consultation du Chatbot RAG

**Priorité:** HAUTE  
**Description:** Poser des questions en langage naturel sur les tickets et obtenir des réponses contextuelles.

**Critères d'acceptation:**
- Interface de chat intuitive
- Réponses basées sur la base de connaissances (tickets historiques)
- Affichage des sources utilisées pour la réponse
- Support du français et de l'arabe
- Historique de conversation conservé durant la session
- Temps de réponse < 5 secondes

#### RF-A-005: Historique des Analyses

**Priorité:** MOYENNE  
**Description:** Consulter l'historique des tickets analysés.

**Critères d'acceptation:**
- Liste des analyses précédentes avec date et heure
- Filtres par catégorie, priorité, date
- Recherche par mots-clés
- Pagination des résultats
- Export en CSV/Excel

#### RF-A-006: Recommandations de Résolution

**Priorité:** MOYENNE  
**Description:** Recevoir des suggestions de résolution basées sur des tickets similaires.

**Critères d'acceptation:**
- Affichage des 3-5 tickets les plus similaires
- Score de similarité affiché
- Résolutions passées proposées
- Liens vers la documentation pertinente

### 3.2 Exigences Fonctionnelles - SUPERVISEUR

#### RF-S-001: Tableau de Bord Analytique

**Priorité:** HAUTE  
**Description:** Vue d'ensemble des performances et statistiques.

**Critères d'acceptation:**
- Nombre total de tickets analysés (jour/semaine/mois)
- Répartition par catégorie (graphique camembert)
- Répartition par priorité (graphique barres)
- Temps de résolution moyen par catégorie
- Tendances temporelles (graphique linéaire)
- KPIs clés en temps réel

#### RF-S-002: Performance des Modèles

**Priorité:** HAUTE  
**Description:** Consulter les métriques de performance des modèles ML.

**Critères d'acceptation:**
- Accuracy, Precision, Recall, F1-Score pour la classification
- RMSE, MAE, R² pour la régression
- Matrice de confusion interactive
- Courbe ROC et AUC
- Évolution des performances dans le temps
- Alertes si performances dégradées

#### RF-S-003: Rapports et Export

**Priorité:** MOYENNE  
**Description:** Générer et exporter des rapports d'analyse.

**Critères d'acceptation:**
- Génération de rapports PDF personnalisés
- Export des données brutes en CSV/Excel
- Rapports programmables (quotidien, hebdomadaire)
- Envoi automatique par email
- Templates de rapports prédéfinis

#### RF-S-004: Gestion des Alertes

**Priorité:** MOYENNE  
**Description:** Configurer des alertes automatiques.

**Critères d'acceptation:**
- Alerte si ticket classé "Urgent" détecté
- Alerte si temps de résolution estimé > seuil
- Alerte si volume de tickets anormal
- Notifications en temps réel dans l'interface
- Notifications par email configurables

### 3.3 Exigences Fonctionnelles - ADMINISTRATEUR

#### RF-AD-001: Gestion des Utilisateurs

**Priorité:** HAUTE  
**Description:** Gérer les comptes utilisateurs de la plateforme.

**Critères d'acceptation:**
- Création, modification, suppression d'utilisateurs
- Attribution des rôles (Agent, Superviseur, Admin)
- Réinitialisation des mots de passe
- Activation/désactivation des comptes
- Logs des actions utilisateurs

#### RF-AD-002: Réentraînement des Modèles

**Priorité:** HAUTE  
**Description:** Mettre à jour les modèles ML avec de nouvelles données.

**Critères d'acceptation:**
- Upload de nouveaux datasets d'entraînement
- Validation automatique du format des données
- Lancement du réentraînement à la demande
- Comparaison ancien vs nouveau modèle
- Rollback possible vers version précédente
- Sauvegarde automatique des modèles

#### RF-AD-003: Gestion de la Base Vectorielle

**Priorité:** HAUTE  
**Description:** Administrer la base de connaissances du RAG.

**Critères d'acceptation:**
- Ajout de nouveaux documents à indexer
- Suppression de documents obsolètes
- Réindexation complète possible
- Statistiques sur la base (nombre de documents, taille)
- Vérification de l'intégrité des embeddings

#### RF-AD-004: Configuration du LLM

**Priorité:** MOYENNE  
**Description:** Paramétrer le modèle de langage.

**Critères d'acceptation:**
- Choix du modèle LLM (OpenAI, Mistral, Ollama)
- Configuration de la température
- Limite de tokens
- Personnalisation du prompt système
- Gestion des clés API

### 3.4 Exigences Fonctionnelles - SYSTÈME

#### RF-SY-001: Prétraitement Automatique des Textes

**Priorité:** HAUTE  
**Description:** Nettoyer et préparer les textes pour l'analyse.

**Critères d'acceptation:**
- Conversion en minuscules
- Suppression des caractères spéciaux inutiles
- Gestion des caractères arabes
- Tokenisation appropriée
- Suppression des stopwords (français et arabe)
- Lemmatisation optionnelle

#### RF-SY-002: Vectorisation TF-IDF

**Priorité:** HAUTE  
**Description:** Transformer les textes en vecteurs numériques.

**Critères d'acceptation:**
- Vectorisation avec TfidfVectorizer
- Limite configurable de features (max_features)
- Support des n-grams (unigrammes, bigrammes)
- Sauvegarde du vectoriseur pour réutilisation
- Cohérence entre entraînement et prédiction

#### RF-SY-003: Mise à Jour des Embeddings

**Priorité:** MOYENNE  
**Description:** Actualiser la base vectorielle avec les nouveaux documents.

**Critères d'acceptation:**
- Détection automatique des nouveaux documents
- Génération des embeddings (SentenceTransformers)
- Insertion dans ChromaDB
- Vérification de la qualité des embeddings
- Logs de mise à jour

---

## 4. EXIGENCES NON-FONCTIONNELLES

### 4.1 Performance

#### RNF-P-001: Temps de Réponse

- Classification d'un ticket : < 2 secondes
- Prédiction temps de résolution : < 1 seconde
- Réponse du chatbot RAG : < 5 secondes
- Chargement du tableau de bord : < 3 secondes
- Recherche dans l'historique : < 1 seconde

#### RNF-P-002: Capacité

- Support de 100 utilisateurs simultanés minimum
- Traitement de 1000 tickets par jour
- Base vectorielle supportant 50 000 documents
- Historique conservé sur 2 ans minimum

#### RNF-P-003: Disponibilité

- Uptime : 99% minimum (hors maintenance planifiée)
- Maintenance planifiée : fenêtres nocturnes uniquement
- Temps de récupération après panne : < 1 heure

### 4.2 Sécurité

#### RNF-S-001: Authentification

- Authentification obligatoire pour l'historique et les fonctions admin
- Mots de passe hashés (minimum 8 caractères)
- Session expirée après 30 minutes d'inactivité
- Protection contre les attaques par force brute

#### RNF-S-002: Autorisation

- Contrôle d'accès basé sur les rôles (RBAC)
- Séparation des privilèges Agent/Superviseur/Admin
- Logs de toutes les actions sensibles

#### RNF-S-003: Protection des Données

- Conformité RGPD et lois marocaines sur les données personnelles
- Anonymisation possible des données de test
- Chiffrement des données sensibles au repos
- HTTPS obligatoire en production

### 4.3 Utilisabilité

#### RNF-U-001: Interface Utilisateur

- Interface responsive (desktop prioritaire)
- Support du français comme langue principale
- Support de l'arabe (RTL) pour les contenus
- Messages d'erreur clairs et actionnables
- Aide contextuelle disponible

#### RNF-U-002: Accessibilité

- Contraste suffisant pour la lisibilité
- Navigation au clavier possible
- Textes alternatifs pour les graphiques
- Taille de police ajustable

### 4.4 Maintenabilité

#### RNF-M-001: Code

- Code commenté en français
- Structure modulaire (séparation ML / RAG / UI)
- Tests unitaires pour les fonctions critiques
- Documentation technique à jour

#### RNF-M-002: Déploiement

- Instructions d'installation claires (README)
- Dépendances gérées via requirements.txt
- Configuration via variables d'environnement
- Scripts de démarrage automatisés

---

## 5. CONTRAINTES TECHNIQUES

### 5.1 Stack Technologique Obligatoire

#### Machine Learning

- **Langage :** Python 3.9+
- **ML Classique :** scikit-learn (TF-IDF, Random Forest, KNN)
- **Boosting :** XGBoost ou LightGBM
- **Évaluation :** scikit-learn metrics

#### RAG / GenAI

- **Embeddings :** sentence-transformers (all-MiniLM-L6-v2)
- **Base Vectorielle :** ChromaDB
- **Framework LLM :** LangChain
- **LLM :** OpenAI API, Mistral API ou Ollama (local)

#### Interface Utilisateur

- **Framework :** Streamlit
- **Visualisation :** Matplotlib, Plotly
- **Composants UI :** Streamlit natif

#### Données

- **Format :** CSV, JSON
- **Stockage modèles :** Joblib (.pkl)
- **Stockage vectoriel :** ChromaDB (fichiers locaux)

### 5.2 Contraintes d'Architecture

#### CT-001: Architecture Modulaire

- Séparation claire entre les modules : `src/ml/`, `src/rag/`, `app/`
- Indépendance des composants ML et RAG
- Interface unifiée via Streamlit

#### CT-002: Portabilité

- Fonctionnement sur Windows, Linux, macOS
- Pas de dépendance à des services cloud obligatoires
- Possibilité de fonctionner en mode local (Ollama)

#### CT-003: Reproductibilité

- Seeds fixés pour les modèles ML
- Versioning des modèles entraînés
- Sauvegarde des hyperparamètres utilisés

---

## 6. RÈGLES MÉTIER

### RM-001: Classification des Tickets

- Un ticket appartient à une seule catégorie principale
- Les catégories possibles sont : Technique, Facturation, Commercial, Réseau, Autre
- Les niveaux de priorité sont : Urgente, Haute, Moyenne, Basse
- Le modèle doit avoir un score de confiance minimum de 60% pour afficher une prédiction

### RM-002: Prédiction du Temps de Résolution

- Le temps est exprimé en heures (0.5h minimum)
- Le temps maximum prédit est plafonné à 168h (1 semaine)
- Si les données sont insuffisantes, afficher "Estimation non disponible"
- L'intervalle de confiance est de ±20% par défaut

### RM-003: Chatbot RAG

- Le chatbot répond uniquement sur la base des documents indexés
- Si aucune information pertinente n'est trouvée, le chatbot l'indique clairement
- Le nombre de documents sources affichés est limité à 3
- Le chatbot ne donne pas de conseils médicaux ou juridiques

### RM-004: Gestion des Langues

- Le français est la langue par défaut de l'interface
- Les tickets peuvent être rédigés en français, arabe classique ou darija
- Le modèle ML est entraîné sur du texte multilingue
- Les réponses du chatbot sont en français sauf si la question est en arabe

### RM-005: Détection des Mots-Clés Urgents

- Mots-clés français urgents : "urgent", "critique", "bloquant", "panne totale"
- Mots-clés arabes urgents : "عاجل", "مهم", "ضروري"
- Détection automatique élève la priorité suggérée

### RM-006: Qualité des Données

- Un ticket doit contenir au minimum 10 mots pour être analysé
- Les tickets trop courts génèrent un avertissement
- Les caractères spéciaux excessifs sont signalés

---

## 7. MATRICE DE TRAÇABILITÉ DES EXIGENCES

| ID Exigence | Priorité | Acteur | Module | Fichier Source | Fonction | Test ID |
|-------------|----------|--------|--------|----------------|----------|---------|
| RF-A-001 | HAUTE | Agent | app/ | main.py | submit_ticket() | TC-001 |
| RF-A-002 | HAUTE | Agent | src/ml/ | train_model.py | predict_class() | TC-002 |
| RF-A-003 | HAUTE | Agent | src/ml/ | regression.py | predict_time() | TC-003 |
| RF-A-004 | HAUTE | Agent | src/rag/ | chatbot.py | ask_bot() | TC-004 |
| RF-A-005 | MOYENNE | Agent | app/ | main.py | show_history() | TC-005 |
| RF-A-006 | MOYENNE | Agent | src/rag/ | chatbot.py | get_similar() | TC-006 |
| RF-S-001 | HAUTE | Superviseur | app/ | main.py | dashboard() | TC-007 |
| RF-S-002 | HAUTE | Superviseur | src/ml/ | evaluate.py | get_metrics() | TC-008 |
| RF-S-003 | MOYENNE | Superviseur | app/ | main.py | export_report() | TC-009 |
| RF-S-004 | MOYENNE | Superviseur | app/ | main.py | configure_alerts() | TC-010 |
| RF-AD-001 | HAUTE | Admin | app/ | admin.py | manage_users() | TC-011 |
| RF-AD-002 | HAUTE | Admin | src/ml/ | train_model.py | retrain_model() | TC-012 |
| RF-AD-003 | HAUTE | Admin | src/rag/ | ingest.py | manage_vectordb() | TC-013 |
| RF-AD-004 | MOYENNE | Admin | src/rag/ | chatbot.py | configure_llm() | TC-014 |
| RF-SY-001 | HAUTE | Système | src/ml/ | clean_data.py | preprocess() | TC-015 |
| RF-SY-002 | HAUTE | Système | src/ml/ | train_model.py | vectorize() | TC-016 |
| RF-SY-003 | MOYENNE | Système | src/rag/ | ingest.py | update_embeddings() | TC-017 |

---

## 8. FONCTIONNALITÉS PAR ORDRE DE PRIORITÉ

### Phase 1 - MVP (Must Have) - Jours 1-6

1. **Préparation des données** (RF-SY-001)
2. **Vectorisation TF-IDF** (RF-SY-002)
3. **Classification automatique** (RF-A-002)
4. **Prédiction temps de résolution** (RF-A-003)
5. **Interface de soumission de ticket** (RF-A-001)
6. **Embeddings et base vectorielle** (RF-SY-003)
7. **Chatbot RAG fonctionnel** (RF-A-004)

### Phase 2 - Fonctionnalités Essentielles (Should Have) - Jours 7-9

1. **Tableau de bord superviseur** (RF-S-001)
2. **Métriques de performance** (RF-S-002)
3. **Historique des analyses** (RF-A-005)
4. **Recommandations de résolution** (RF-A-006)

### Phase 3 - Fonctionnalités Avancées (Could Have) - Jours 10-11

1. **Export et rapports** (RF-S-003)
2. **Système d'alertes** (RF-S-004)
3. **Gestion des utilisateurs** (RF-AD-001)
4. **Réentraînement des modèles** (RF-AD-002)

### Phase 4 - Finalisation (Won't Have - hors scope initial)

1. Configuration avancée du LLM (RF-AD-004)
2. Application mobile
3. Intégration avec systèmes de ticketing existants (Zendesk, Freshdesk)
4. API REST pour intégration tierce

---

## 9. RISQUES ET MITIGATION

### Risque 1: Qualité insuffisante du dataset

- **Impact:** ÉLEVÉ
- **Probabilité:** MOYENNE
- **Mitigation:** Utiliser un dataset hybride (Kaggle + données fictives marocaines), valider manuellement un échantillon

### Risque 2: Overfitting des modèles ML

- **Impact:** ÉLEVÉ
- **Probabilité:** MOYENNE
- **Mitigation:** Cross-validation, régularisation, split train/validation/test rigoureux

### Risque 3: Hallucinations du chatbot LLM

- **Impact:** MOYEN
- **Probabilité:** HAUTE
- **Mitigation:** Prompt engineering strict, affichage systématique des sources, limite au contexte récupéré

### Risque 4: Temps de réponse lent du RAG

- **Impact:** MOYEN
- **Probabilité:** MOYENNE
- **Mitigation:** Optimisation des embeddings, limitation du nombre de documents récupérés, caching

### Risque 5: Problèmes de compatibilité linguistique (arabe/darija)

- **Impact:** MOYEN
- **Probabilité:** MOYENNE
- **Mitigation:** Tests approfondis avec textes multilingues, modèle d'embedding supportant l'arabe

### Risque 6: Dépassement du délai de 12 jours

- **Impact:** ÉLEVÉ
- **Probabilité:** MOYENNE
- **Mitigation:** Priorisation stricte MVP, répartition claire des tâches, points quotidiens

---

## 10. GLOSSAIRE

- **Ticket Support :** Demande d'assistance client contenant une description textuelle d'un problème
- **Classification :** Attribution automatique d'une catégorie à un ticket basée sur son contenu
- **Régression :** Prédiction d'une valeur numérique continue (temps de résolution)
- **TF-IDF :** Term Frequency-Inverse Document Frequency, méthode de vectorisation de texte
- **Random Forest :** Algorithme d'apprentissage supervisé basé sur des arbres de décision
- **XGBoost :** Algorithme de boosting pour la classification et la régression
- **RAG :** Retrieval-Augmented Generation, technique combinant recherche et génération de texte
- **Embedding :** Représentation vectorielle dense d'un texte capturant son sens sémantique
- **ChromaDB :** Base de données vectorielle open-source pour stocker et rechercher des embeddings
- **LLM :** Large Language Model, modèle de langage de grande taille (GPT, Mistral, etc.)
- **LangChain :** Framework Python pour construire des applications basées sur les LLM
- **Streamlit :** Framework Python pour créer des applications web interactives
- **Darija :** Dialecte arabe marocain parlé au quotidien
- **F1-Score :** Métrique d'évaluation combinant précision et rappel
- **RMSE :** Root Mean Square Error, mesure d'erreur pour la régression
- **Matrice de Confusion :** Tableau montrant les prédictions correctes et incorrectes par classe

---

## 11. VALIDATION ET APPROBATION

| Rôle | Nom | Date | Signature |
|------|-----|------|-----------|
| Chef de Projet | _________________ | ____/____/2026 | _________________ |
| Expert ML (Personne A) | _________________ | ____/____/2026 | _________________ |
| Expert RAG (Personne B) | _________________ | ____/____/2026 | _________________ |
| Intégrateur (Personne C) | _________________ | ____/____/2026 | _________________ |
| Encadrant Académique | _________________ | ____/____/2026 | _________________ |

---

## 12. HISTORIQUE DES RÉVISIONS

| Version | Date | Auteur | Description des modifications |
|---------|------|--------|-------------------------------|
| 1.0 | 31/01/2026 | Équipe Projet | Version initiale du document |
| | | | |
| | | | |

---

**Date de prochaine révision :** À définir après validation de la version 1.0  
**Contact :** Équipe Projet - Plateforme d'Analyse Intelligente de Tickets Support  
**Localisation :** Maroc

---

*Document confidentiel - Usage interne uniquement*
