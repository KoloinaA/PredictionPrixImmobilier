import pandas as pd;

df= pd.read_csv("train.csv")

#ne garder que les colonnes numériques
num_df= df.select_dtypes(include=["int64","float64"])
corr = num_df.corr()

#afficher les colonnes à forte correlation avec SalePrice
#print(corr["SalePrice"].sort_values(ascending=False))

#sauvegarde dans un fichier des variables explicatives à forte correlation avec SalePrice
with open("correlation_resultat.txt", "w") as f:
    print(corr, file=f)