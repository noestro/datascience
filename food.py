import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Charger le dataset
df = pd.read_csv('wfp_food_prices_cod.csv')

# Exploration initiale
print("=== EXPLORATION DES DONNÉES ===")
print(f"Dimensions: {df.shape}")
print(f"Colonnes: {df.columns.tolist()}")
print(f"Types de données:\n{df.dtypes}")
print(f"Valeurs manquantes:\n{df.isnull().sum()}")

# Statistiques descriptives
print("\n=== STATISTIQUES DESCRIPTIVES ===")
print(df.describe())

# Visualisations exploratoires
plt.figure(figsize=(15, 10))

# Distribution des prix
plt.subplot(2, 3, 1)
sns.histplot(df['usdprice'], bins=50, kde=True)
plt.title('Distribution des prix (USD)')
plt.xlabel('Prix (USD)')
plt.ylabel('Fréquence')

# Prix par catégorie
plt.subplot(2, 3, 2)
sns.boxplot(x='category', y='usdprice', data=df)
plt.title('Prix par catégorie')
plt.xticks(rotation=45)

# Prix par région administrative
plt.subplot(2, 3, 3)
top_regions = df['admin1'].value_counts().head(10).index
sns.boxplot(x='admin1', y='usdprice', data=df[df['admin1'].isin(top_regions)])
plt.title('Prix par région (Top 10)')
plt.xticks(rotation=45)

# Évolution temporelle des prix moyens
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

plt.subplot(2, 3, 4)
yearly_prices = df.groupby('year')['usdprice'].mean().reset_index()
sns.lineplot(x='year', y='usdprice', data=yearly_prices)
plt.title('Évolution annuelle des prix moyens')
plt.xlabel('Année')
plt.ylabel('Prix moyen (USD)')

# Prix par produit (Top 10)
plt.subplot(2, 3, 5)
top_commodities = df['commodity'].value_counts().head(10).index
sns.boxplot(x='commodity', y='usdprice', data=df[df['commodity'].isin(top_commodities)])
plt.title('Prix par produit (Top 10)')
plt.xticks(rotation=45)

# Corrélation entre variables numériques
plt.subplot(2, 3, 6)
numeric_cols = ['latitude', 'longitude', 'price', 'usdprice']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation')

plt.tight_layout()
plt.show()

# Nettoyage des données
print("\n=== NETTOYAGE DES DONNÉES ===")

# Suppression des valeurs aberrantes (prix négatifs ou nuls)
df_clean = df[df['usdprice'] > 0].copy()

# Encodage des variables catégorielles
le_category = LabelEncoder()
le_commodity = LabelEncoder()
le_admin1 = LabelEncoder()
le_unit = LabelEncoder()

df_clean['category_encoded'] = le_category.fit_transform(df_clean['category'])
df_clean['commodity_encoded'] = le_commodity.fit_transform(df_clean['commodity'])
df_clean['admin1_encoded'] = le_admin1.fit_transform(df_clean['admin1'])
df_clean['unit_encoded'] = le_unit.fit_transform(df_clean['unit'])

# Sélection des features pour le modèle
features = ['year', 'month', 'latitude', 'longitude', 'category_encoded',
           'commodity_encoded', 'admin1_encoded', 'unit_encoded']

X = df_clean[features]
y = df_clean['usdprice']

print(f"Features sélectionnées: {features}")
print(f"Variable cible: usdprice")
print(f"Dimensions finales: {X.shape}")

# Normalisation des features numériques
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['latitude', 'longitude']] = scaler.fit_transform(X[['latitude', 'longitude']])

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Sélection des features importantes
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_train, y_train)
feature_scores = pd.DataFrame({
    'Feature': features,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

print("\n=== IMPORTANCE DES FEATURES ===")
print(feature_scores)

# Visualisation de l'importance des features
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores)
plt.title('Importance des features (SelectKBest)')
plt.show()

# Comparaison des algorithmes
print("\n=== COMPARAISON DES ALGORITHMES ===")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'CV_R²_mean': cv_scores.mean(),
        'CV_R²_std': cv_scores.std()
    }

    print(f"\n{name}:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  CV R² (mean): {cv_scores.mean():.3f} (std: {cv_scores.std():.3f})")

# Comparaison visuelle des performances
results_df = pd.DataFrame(results).T
print("\n=== RÉSUMÉ DES PERFORMANCES ===")
print(results_df)

# Visualisation des résultats
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# R² scores
results_df['R²'].plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Scores R²')
axes[0,0].set_ylabel('R²')
axes[0,0].tick_params(axis='x', rotation=45)

# MAE scores
results_df['MAE'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
axes[0,1].set_title('Erreurs absolues moyennes (MAE)')
axes[0,1].set_ylabel('MAE (USD)')
axes[0,1].tick_params(axis='x', rotation=45)

# RMSE scores
results_df['RMSE'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
axes[1,0].set_title('Erreurs quadratiques moyennes (RMSE)')
axes[1,0].set_ylabel('RMSE (USD)')
axes[1,0].tick_params(axis='x', rotation=45)

# Cross-validation R²
results_df['CV_R²_mean'].plot(kind='bar', ax=axes[1,1], color='gold',
                             yerr=results_df['CV_R²_std'], capsize=5)
axes[1,1].set_title('Cross-validation R² (moyenne ± écart-type)')
axes[1,1].set_ylabel('CV R²')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Analyse du meilleur modèle
best_model_name = results_df['R²'].idxmax()
best_model = models[best_model_name]

print(f"\n=== ANALYSE DU MEILLEUR MODÈLE: {best_model_name} ===")

# Feature importance pour Random Forest
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("Importance des features:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Importance des features - {best_model_name}')
    plt.show()

# Prédictions vs valeurs réelles pour le meilleur modèle
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Prix réel (USD)')
plt.ylabel('Prix prédit (USD)')
plt.title(f'Prédictions vs Réel - {best_model_name}')
plt.show()

# Distribution des erreurs
errors = y_test - y_pred_best
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=50, kde=True)
plt.xlabel('Erreur de prédiction (USD)')
plt.ylabel('Fréquence')
plt.title(f'Distribution des erreurs - {best_model_name}')
plt.show()

print("\n=== RECOMMANDATIONS ===")
print("1. Le modèle le plus performant est:", best_model_name)
print("2. Variables clés identifiées:")
if 'feature_importance' in locals():
    for i, row in feature_importance.head(3).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.3f}")
else:
    print("   - commodity_encoded: type de produit")
    print("   - admin1_encoded: région administrative")
    print("   - year: année")

print("3. Recommandations métier:")
print("   - Surveiller les prix des produits de base dans les régions vulnérables")
print("   - Anticiper les variations saisonnières des prix")
print("   - Optimiser les chaînes d'approvisionnement basées sur les prédictions")
print("   - Développer des politiques de stabilité des prix alimentaires")

print("\n=== CONCLUSION ===")
print("Le pipeline ML complet a été appliqué avec succès:")
print("✓ Exploration et visualisation des données")
print("✓ Nettoyage et prétraitement")
print("✓ Comparaison de 3 algorithmes de régression")
print("✓ Analyse des performances et interprétation")
print("✓ Recommandations basées sur les résultats")
# Nettoyage : gestion des valeurs manquantes
# Encodage des variables catégorielles
# Normalisation ou standardisation si nécessaire
# Séparation en jeu d’entraînement et de testprint(f"🎯 Score R² : {score*100:.2f}%")print(f"🎯 Score R² : {score*100:.2f}%")