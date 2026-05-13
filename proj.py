import pandas as pd
#import numpy as np
#mport matplotlib.pyplot as plt

#data = {"Groupe": ["X","Y","X","Y","X"], "Score": [4, 7, 5, 6, 8]}
#df = pd.DataFrame(data)


#df_grouped= df.groupby("Groupe") ["Score"].mean() #Calculez le score moyen par groupe.
#print(df_grouped)

#Jointure de DataFrames. Soient deux DataFrames
'''left =pd.DataFrame({"ID":[1,2,3],"Nom":["A","B","C"]})
right = pd.DataFrame({"ID":[2,3,4], "ville":["X","Y","Z"]})
#Fusionnez-les sur la colonne ID avec how='inner'
merged= pd.merge(left,right , on="ID" , how ="inner")
print(merged)

df = pd.DataFrame({"Nom": ["A", "B", "C"], "Age": [25, 30, 22]})
print(df)
print(type(df))  # <class 'pandas.core.frame.DataFrame'>
print(df["Age"])  # renvoie une Series

df = pd.DataFrame({"Valeur": [10, -5, 7, -2, 0]})
df_pos = df[df["Valeur"] >= 0]
print(df_pos)
df=pd.DataFrame({"A":[1,None,3], "B":[4,5,None]})
df_clean=df.dropna().reset_index(drop=True)
print(df_clean)

x = [0,1,2,3]
y = [2*xi + 1 for xi in x]
plt.plot(x,y, marker='o', linestyle='-')
plt.title("Linéaire: y=2x+1")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
#nuage des points
x= np.random.rand(50)
y=np.random.rand(50)
plt.scatter(x,y ,s=100*np.random.rand(50), c=np.random.rand(50),alpha=0.5)
plt.show()

#charge les data
data={
    'date':['2023-01-01','2023-01-01','2023-01-01','2023-01-02','2023-01-02'],
    'produit':['produitA','produitB','produitC','produitA','produitB'],
    'region':['sud','nord','nord','ouest','ouest'],
    'ventes':[1844,1427,2713,2677,2000]
}

df=pd.DataFrame(data)
print(df)

plt.figure(figsize=(10,6))
plt.hist(df['ventes'], bins=20, color='skyblue', edgecolor='black')

plt.title('distribution des ventes', fontsize=14)
plt.xlabel('Montant des ventes', fontsize=12)
plt.ylabel("frequences", fontsize=12)
plt.grid(axis="y" , alpha=0.4)
plt.show()   

x=[1,2,3,]
y=[1,4,9]
plt.subplot(1,2,1)#1 ligne, 2 colonnes, 1er graphique
plt.plot(x,y,c='red',lw=3,ls='--',label='quadrqtique')
plt.subplot(1,2,2)#1 ligne, 2 colonnes, 2eme graphique
plt.plot(x,y,c='black',lw=3,ls='--',label='quadrqtique')
plt.show()

x=np.linspace(0,2,10)
y=np.linspace(0,2,10)
y=x**2
print(x)
print(y)
plt.plot(x,y,c='red',lw=3,ls='--',label='quadratique')
plt.plot(x,label='cubique') #2*2*2=8 dans l'affichage 
#plt.figure(figsize=(2,2))
#plt.grid(True)
plt.xlabel('xlabel') 
plt.ylabel('ylabel')
plt.title('big_boss')
plt.legend()
plt.show()

x = [0,1,2,3]
y1 = [xi**2 for xi in x]
y2 = [xi**3 for xi in x]
plt.plot(x, y1, 'r--', label='x²')
plt.plot(x, y2, 'b-.', label='x³')
plt.legend()
plt.title("Comparaison de x² et x³")
plt.xlabel("x")
plt.ylabel("y")
plt.show()'''

data={'age':[25,30,35,20,40,20]}
df = pd.DataFrame(data)
moyenne = df['age'].mean()
mediane = df['age'].median()
ecart_type = df['age'].std()
print(f'Moyenne: {moyenne}, Mediane: {mediane}, Ecart_type: {ecart_type}')
