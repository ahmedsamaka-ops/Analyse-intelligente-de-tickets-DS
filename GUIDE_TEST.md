# 🎯 Guide de Test du Modèle - Analyse Intelligente de Tickets

## Prérequis

### 1. Installer Python
Télécharger Python 3.10+ : https://www.python.org/downloads/

### 2. Installer les dépendances
Ouvrir un terminal dans le dossier du projet et taper :
```bash
pip install -r requirements.txt
```

---

## 🚀 Tester le Modèle

### Option 1 : Test Interactif (Recommandé)

```bash
python test_interactif.py
```

Ensuite, tape tes tickets :
```
============================================================
   TEST INTERACTIF DES MODELES
   Tape 'quit' pour quitter
============================================================

Tape ton ticket : problème connexion wifi
  → Catégorie : Réseau & Connexion
  → Urgence   : Basse

Tape ton ticket : création compte AD pour ahmed
  → Catégorie : Gestion Comptes AD
  → Urgence   : Basse

Tape ton ticket : quit
Au revoir!
```

### Option 2 : Test Simple

1. Ouvrir `test_model.py`
2. Modifier la ligne 55 :
```python
texte_test = "ton ticket ici"
```
3. Exécuter :
```bash
python test_model.py
```

---

## 📝 Exemples de Tickets à Tester

| Ticket | Catégorie Attendue | Urgence |
|--------|-------------------|---------|
| `problème connexion wifi maphoffice` | Réseau & Connexion | Basse |
| `demande création compte AD pour sarah` | Gestion Comptes AD | Basse |
| `imprimante bloquée service export` | Impressions & Scanner | Basse |
| `accès au partage qualité` | Accès & Partages | Basse |
| `problème SAP bloqué urgent` | Applications & SAP | Moyenne |
| `réinitialisation mot de passe` | Sécurité & MDP | Basse |
| `pb VPN connexion impossible` | Réseau & Connexion | Basse |
| `laptop ne démarre plus` | Matériel | Basse |

---

## 📊 Les 11 Catégories Disponibles

1. **Gestion Comptes AD** - Création, désactivation comptes
2. **Accès & Partages** - Dossiers partagés, permissions
3. **Réseau & Connexion** - Wifi, VPN, internet
4. **Impressions & Scanner** - Imprimantes, scanners
5. **Applications & SAP** - SAP, logiciels métier
6. **Matériel** - PC, laptop, écran
7. **Téléphonie** - Lignes téléphoniques
8. **Sécurité & MDP** - Mots de passe, antivirus
9. **Projets & Dev** - Création projets Citrix
10. **Messagerie** - Outlook, email
11. **Autre** - Reste

---

## 📈 Performances du Modèle

| Tâche | Score |
|-------|-------|
| **Catégorie** | 73.91% accuracy |
| **Urgence** | 68.64% F1-macro |

---

## ❓ Problèmes Fréquents

### "Module not found"
```bash
pip install scikit-learn pandas joblib
```

### "File not found: models/..."
Vérifier que tu es dans le bon dossier :
```bash
cd "C:\chemin\vers\Analyse_intelligente_de_tickets_DS"
```

---

## 📁 Structure du Projet

```
Analyse_intelligente_de_tickets_DS/
├── test_interactif.py     ← LANCER CECI POUR TESTER
├── test_model.py          ← Test simple
├── models/                ← Modèles entraînés
├── data/                  ← Données
├── src/ml/                ← Scripts d'entraînement
└── docs/                  ← Documentation
```

---

**Bon test ! 🚀**
