import pandas as pd

# charger du dataset
df = pd.read_csv('pandas_sales_dataset.csv')
df["date"] = pd.to_datetime(df["date"])
df["montant_brut"] = df["quantite"] * df["prix_unitaire"]
df["montant_net"] = df["montant_brut"] * (1 - df["remise"])

# explorer le dataset
# print(df.head())
# print(df.dtypes)
# print(df.shape)

# filtrer les ventes via le canal web
vente_web = df[df["canal"] == "Web"]
print(vente_web)

# filtrer les grosses ventes (montant net > 2000)
grosses_ventes = df[df["montant_net"] > 2000]
print(grosses_ventes)

# selection colonnes
ventes_selection = vente_web[["date", "client", "produit", "montant_net"]]
print(ventes_selection)

# trier les ventes par montant net décroissant
ventes_triees = df.sort_values(by="montant_net", ascending=False)
print(ventes_triees)

# benefice estime 25% du montant net
df["benefice_estime"] = df["montant_net"] * 0.25
print(df[["date", "client", "produit", "montant_net", "benefice_estime"]])

# somme totale de montant net
somme_montant_net = df["montant_net"].sum()
print(f"Somme totale du montant net: {somme_montant_net}") 
