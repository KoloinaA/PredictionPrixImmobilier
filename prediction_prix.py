import pandas as pd;
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

train_data= pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Supposons ceci comme colonnes pertinentes
# features = ["Street","Neighborhood","Electrical","TotRmsAbvGrd"]
# target = "SalePrice"

# x_train = train_data[features]
# y_train = train_data[target]

# x_test = test_data[features]

# x_train = x_train.fillna(0)
# x_test = x_test.fillna(0)

# # print(x_train.dtypes)
# #mampitovy ny variable ho type entier daholo
# x_train = pd.get_dummies(x_train)
# x_test = pd.get_dummies(x_test)

# # Aligner les colonnes entre train et test
# x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)
target = "SalePrice"

# 2. On sépare les features et la target
#on enleve SalePrice pour ne garder que les variables explicatives
X = train_data.drop(columns=[target])
y = train_data[target]

# 3. On identifie les colonnes numériques et catégorielles
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# 4. Prétraitement : Imputer et encoder
#SimpleImputer remplace la valeur manquante (ici par la mediane)
#OnHotEncoder transforme les valeurs en binaire
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# 5. Pipeline de transformation uniquement pour les données
X_processed = preprocessor.fit_transform(X)
#application de la meme transformation pour les données de test
X_test_processed = preprocessor.transform(test_data)

# 6. Sélection automatique des meilleures features (ex : les 50 meilleures)
selector = SelectKBest(score_func=f_regression, k=50)
X_selected = selector.fit_transform(X_processed, y)
X_test_selected = selector.transform(X_test_processed)

model = LinearRegression()
# model.fit(x_train, y_train)
model.fit(X_selected, y)

# # Prédiction
y_pred = model.predict(X_test_selected)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice" : y_pred
})

submission.to_csv("submission.csv", index= False)

#sauvegarde du modele, preprocessor, selector
joblib.dump(model, "model_prix_immobilier")
joblib.dump(preprocessor,"preprocessor.pkl")
joblib.dump(selector,"selector.pkl")