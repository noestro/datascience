import pandas as pd
import numpy as np
#analyse statistique
'''data={'age':[25,30,35,20,40,20]}
df = pd.DataFrame(data)
moyenne = df['age'].mean()
mediane = df['age'].median()
ecart_type = df['age'].std()
print(f'Moyenne: {moyenne}, Mediane: {mediane}, Ecart_type: {ecart_type}')

#modelisation(refression lineaire)
from sklearn.linear_model import LinearRegression

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2,4,6,8,10])
model = LinearRegression().fit(x,y)
print(f'coefficient :{model.coef_}')'''

#nettoyage et preparation(remplacer valeurs manquqntes)

df = pd.DataFrame({'A': [1, 2, np.nan, 4]})
df['A'].fillna(df['A'].mean(), inplace=True)
print(df)
#fouille de donnees(association avec mlxtend)
from mlxtend.frequent_patterns import apriori

data = pd.DataFrame({'Pain': [1, 1, 0, 1], 'Beurre': [1, 0, 0, 1]})
frequent_itemsets = apriori(data, min_support=0.5)
print(frequent_itemsets)