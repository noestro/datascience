import pandas as pd
import numpy as np


# DataFrame clients
clients = pd.DataFrame({
    'client_id': [101, 102, 103, 104, 105],
    'nom': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
    'ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse', None],
    'membre_depuis': ['2020-01-15', '2019-05-22', None, '2021-11-30', '2022-03-10']
})

# DataFrame commandes
commandes = pd.DataFrame({
    'commande_id': [1, 2, 3, 4, 5, 6],
    'client_id': [101, 102, 101, 104, 103, 106],
    'produit': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone', 'Monitor'],
    'montant': [1200, 800, 350, 1100, 750, 300],
    'date': ['2023-01-10', '2023-01-12', '2023-01-15', '2023-02-01', '2023-02-05', '2023-02-10']
})

# 1. Nettoyage
clients_clean = clients.dropna(subset=['nom']) # suppresion
clients['ville'] = clients['ville'].fillna('Inconnue') # remplacement
clients['membre_depuis'] = clients['membre_depuis'].fillna(clients['membre_depuis'].max())

# 2. Tri
commandes_triees = commandes.sort_values('montant', ascending=False)

# 3. Aggregation
stats_clients = commandes.groupby('client_id').agg(
    nombre_commandes = ('commande_id', 'count'),
    montant_total = ('montant', 'sum'),
    montant_moyen = ('montant', 'mean')
).reset_index()

# 4. Fusion
rapport = pd.merge(clients, stats_clients, on='client_id', how='left')

# 5. Analyse
parisiens = rapport[rapport['ville'] == 'Paris'][['nom', 'montant_total']]
ventes_par_produit = commandes.groupby('produit')['montant'].sum().reset_index()


# Affichage des resultatss
print("Clients nettoyés :\n", clients_clean)
print("\nCommandes triées :\n", commandes_triees)
print(clients)
print("\nStats des cliens :\n", stats_clients)
print("\nRapport complet :\n", rapport)
print("\nClients parisiens :\n", parisiens)
print("\nVentes par produit :\n", ventes_par_produit)