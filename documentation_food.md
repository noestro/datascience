# Documentation du Programme `food.py`

## Introduction

Le programme `food.py` est un script Python conçu pour analyser et prédire les prix alimentaires à Kinshasa, en République Démocratique du Congo, en utilisant des données réelles provenant du fichier CSV `wfp_food_prices_cod.csv`. Ce fichier contient des données sur les prix des produits alimentaires collectées par le Programme Alimentaire Mondial (WFP) et disponibles sur la plateforme Humanitarian Data Exchange (HDX).

Le programme effectue les étapes suivantes :
- Chargement et exploration des données.
- Filtrage des données pour la région de Kinshasa.
- Préparation des données pour l'analyse (conversion des dates, création de colonnes dérivées).
- Calcul et visualisation de l'évolution des prix moyens annuels.
- Construction et entraînement d'un modèle d'intelligence artificielle (IA) pour prédire les prix futurs.
- Évaluation des performances du modèle.

Ce programme est utile pour les analystes, les décideurs politiques ou les organisations humanitaires qui souhaitent comprendre les tendances des prix alimentaires et anticiper les fluctuations potentielles.

## Utilisation du Programme

### Prérequis
- Python 3.x installé.
- Bibliothèques nécessaires : `pandas`, `matplotlib`, `scikit-learn`.
- Le fichier de données `wfp_food_prices_cod.csv` doit être présent dans le même répertoire que le script.

### Installation des Dépendances
Si les bibliothèques ne sont pas installées, exécutez :
```
pip install pandas matplotlib scikit-learn
```

### Exécution
1. Placez le fichier `wfp_food_prices_cod.csv` dans le répertoire du script.
2. Ouvrez un terminal et naviguez vers le répertoire contenant `food.py`.
3. Activez l'environnement virtuel si nécessaire (par exemple, `source projet_env/bin/activate`).
4. Exécutez le script : `python food.py`.

Le programme affichera des informations dans la console et générera un graphique.

### Sortie Attendue
- Affichage des premières lignes du dataset.
- Liste des colonnes, marchés, produits et période couverte.
- Données filtrées pour Kinshasa.
- Prix moyen par année.
- Graphique de l'évolution des prix.
- Métriques de performance du modèle IA (Score R² et Erreur moyenne).

## Explication du Code, Ligne par Ligne

Voici une analyse détaillée de chaque section du code, avec son utilité et les variables impliquées.

### 1. Importation des Bibliothèques
```python
import pandas as pd
```
- **Utilité** : Importe la bibliothèque `pandas` pour la manipulation des données. `pandas` est essentiel pour lire les fichiers CSV et travailler avec des DataFrames.
- **Variables** : Aucune nouvelle variable créée ici.

### 2. Chargement des Données
```python
df_hdx = pd.read_csv('wfp_food_prices_cod.csv')
```
- **Utilité** : Lit le fichier CSV et le charge dans un DataFrame `pandas`. Cela permet d'accéder aux données sous forme de tableau.
- **Variables** :
  - `df_hdx` : DataFrame contenant toutes les données du fichier CSV. Chaque ligne représente un relevé de prix pour un produit, un marché et une date donnée.

### 3. Exploration Initiale des Données
```python
print(df_hdx.head())
print(df_hdx.columns)
print("Marchés disponibles :")
print(df_hdx['market'].unique())
print("\nProduits disponibles :")
print(df_hdx['commodity'].unique())
print(f"\nPériode : de {df_hdx['date'].min()} à {df_hdx['date'].max()}")
```
- **Utilité** : Affiche un aperçu des données pour vérifier leur intégrité et comprendre la structure. Cela aide à identifier les marchés, produits et période couverte.
- **Variables** :
  - `df_hdx['market']` : Colonne contenant les noms des marchés (villes).
  - `df_hdx['commodity']` : Colonne contenant les noms des produits alimentaires.
  - `df_hdx['date']` : Colonne contenant les dates des relevés.

### 4. Filtrage pour Kinshasa
```python
df_kin = df_hdx[df_hdx['admin1'] == 'Kinshasa'].copy()
print(f"Nombre de relevés à Kinshasa : {len(df_kin)}")
print("\nProduits suivis à Kinshasa :")
print(df_kin['commodity'].unique())
```
- **Utilité** : Filtre les données pour ne conserver que celles relatives à Kinshasa. Cela réduit le dataset à la région d'intérêt et permet une analyse ciblée.
- **Variables** :
  - `df_kin` : Nouveau DataFrame filtré contenant uniquement les données de Kinshasa.
  - `df_hdx['admin1']` : Colonne indiquant la région administrative (ici, 'Kinshasa').

### 5. Préparation des Données pour l'IA
```python
df_kin['date'] = pd.to_datetime(df_kin['date'])
df_kin['Annee'] = df_kin['date'].dt.year
df_kin['Mois'] = df_kin['date'].dt.month
```
- **Utilité** : Convertit la colonne 'date' en format datetime et extrait l'année et le mois. Cela facilite l'analyse temporelle et la création de features pour le modèle IA.
- **Variables** :
  - `df_kin['date']` : Colonne convertie en datetime.
  - `df_kin['Annee']` : Nouvelle colonne contenant l'année extraite.
  - `df_kin['Mois']` : Nouvelle colonne contenant le mois extrait.

### 6. Calcul du Prix Moyen par Année
```python
prix_annuel = df_kin.groupby('Annee')['price'].mean().reset_index()
print(prix_annuel)
```
- **Utilité** : Calcule le prix moyen des produits par année. Cela donne une vue d'ensemble des tendances annuelles des prix.
- **Variables** :
  - `prix_annuel` : DataFrame avec les colonnes 'Annee' et 'price' (prix moyen).
  - `df_kin['price']` : Colonne contenant les prix individuels.

### 7. Visualisation de l'Évolution des Prix
```python
import matplotlib.pyplot as plt 

plt.figure(figsize=(12,6))
plt.plot(prix_annuel['Annee'], prix_annuel['price'], marker='o', color='red')
plt.title("Évolution des prix alimentaires à Kinshasa (Données réelles HDX)")
plt.xlabel("Année")
plt.ylabel("Prix moyen (CDF)")
plt.grid(True)
plt.show()
```
- **Utilité** : Crée un graphique linéaire montrant l'évolution des prix moyens annuels. Cela aide à visualiser les tendances et identifier les périodes de hausse ou baisse des prix.
- **Variables** : Utilise `prix_annuel['Annee']` et `prix_annuel['price']` pour tracer le graphique. `plt` est l'alias pour `matplotlib.pyplot`.

### 8. Création du Modèle IA de Prédiction
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df_kin_sorted = df_kin.sort_values('date')
df_kin_sorted['prix_mois_suivant'] = df_kin_sorted['price'].shift(-1)
df_kin_sorted = df_kin_sorted.dropna()

X = df_kin_sorted[['Annee', 'Mois', 'price']]
y = df_kin_sorted['prix_mois_suivant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
erreur = mean_absolute_error(y_test, predictions)

print(f"🎯 Score R² : {score*100:.2f}%")
print(f"📊 Erreur moyenne : {erreur:.2f} CDF")
print(f"\n🚀 Ton modèle IA avec des VRAIES données est prêt !")
```
- **Utilité** : Construit un modèle de régression utilisant Random Forest pour prédire le prix du mois suivant basé sur l'année, le mois et le prix actuel. Évalue les performances avec R² et l'erreur absolue moyenne.
- **Variables** :
  - `df_kin_sorted` : DataFrame trié par date.
  - `df_kin_sorted['prix_mois_suivant']` : Nouvelle colonne créée en décalant la colonne 'price' d'une ligne (prix du mois suivant).
  - `X` : Features (variables indépendantes) : 'Annee', 'Mois', 'price'.
  - `y` : Variable cible : 'prix_mois_suivant'.
  - `X_train`, `X_test`, `y_train`, `y_test` : Ensembles d'entraînement et de test.
  - `model` : Instance du modèle RandomForestRegressor.
  - `predictions` : Prédictions sur l'ensemble de test.
  - `score` : Score R² (coefficient de détermination).
  - `erreur` : Erreur absolue moyenne en CDF.

## Liste Complète des Variables

- `df_hdx` : DataFrame principal contenant toutes les données du CSV.
- `df_kin` : DataFrame filtré pour Kinshasa.
- `df_kin['date']` : Colonne de dates convertie en datetime.
- `df_kin['Annee']` : Colonne des années extraites.
- `df_kin['Mois']` : Colonne des mois extraits.
- `prix_annuel` : DataFrame des prix moyens par année.
- `plt` : Alias pour matplotlib.pyplot.
- `df_kin_sorted` : DataFrame trié par date.
- `df_kin_sorted['prix_mois_suivant']` : Prix du mois suivant.
- `X` : Matrice des features.
- `y` : Vecteur des cibles.
- `X_train`, `X_test`, `y_train`, `y_test` : Données d'entraînement et de test.
- `model` : Modèle IA entraîné.
- `predictions` : Prédictions du modèle.
- `score` : Score R² du modèle.
- `erreur` : Erreur moyenne absolue.

## Prise de Décision Après les Résultats

Après l'exécution du programme, analysez les résultats pour prendre des décisions informées :

1. **Interprétation des Données Exploratoires** :
   - Si les prix moyens annuels montrent une tendance à la hausse, considérez des interventions pour stabiliser les prix (subventions, importations).
   - Identifiez les produits les plus volatiles et priorisez leur surveillance.

2. **Évaluation du Modèle IA** :
   - Un Score R² élevé (>80%) indique un bon ajustement du modèle. Utilisez-le pour des prédictions fiables.
   - Une Erreur moyenne faible (<1000 CDF) suggère des prédictions précises. Sinon, améliorez le modèle en ajoutant plus de features (ex. : inflation, événements climatiques).
   - Si les performances sont insuffisantes, essayez d'autres algorithmes (ex. : Linear Regression, LSTM pour séries temporelles).

3. **Actions Recommandées** :
   - **Court terme** : Utilisez les prédictions pour ajuster les stocks ou les prix de vente.
   - **Moyen terme** : Intégrez le modèle dans un système de surveillance continue.
   - **Long terme** : Collectez plus de données ou collaborez avec des experts pour affiner les prédictions.
   - Si les données sont insuffisantes, demandez des mises à jour du dataset HDX.

4. **Risques et Limites** :
   - Les prédictions sont basées sur des données historiques ; elles ne tiennent pas compte d'événements imprévus (conflits, catastrophes).
   - Validez toujours les prédictions avec des experts locaux avant de prendre des décisions majeures.

Ce document couvre l'ensemble du programme. Pour toute modification ou extension, consultez la documentation des bibliothèques utilisées (pandas, scikit-learn, matplotlib).
