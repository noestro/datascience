import numpy as np
import pandas as pd
import random

np.random.seed(2)
random.seed(42)
#nombres des lignes
n = 100
#generation aleatoire
data={
    "EleveID":list(range(1,n+1)),
    "Notes_Maths":[random.choice([round(random.uniform(0, 20), 1),np.nan]) for _ in range(n)],
    "Notes_francais":[random.choice([round(random.uniform(0, 20), 1),np.nan]) for _ in range(n)],
    "Absences":[random.randint(0, 20) for _ in range(n)],
    "statut_social":[random.choice(["Favorise", "Defavorise"]) for _ in range(n)]
}

#creation de dataframe 
df = pd.DataFrame(data)
print(df.describe())
#verification d NAN
print("valeurs manquantes par colonnes:")
print(df.isnull().sum())