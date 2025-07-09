import pandas as pd;
from sklearn.linear_model import LinearRegression
train_data= pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Supposons ceci comme colonnes pertinentes
features = ["Street","Neighborhood","Electrical","TotRmsAbvGrd"]
target = "SalePrice"

x_train = train_data[features]
y_train = train_data[target]

x_test = test_data[features]

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

# print(x_train.dtypes)
#mampitovy ny variable ho type entier daholo
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# Aligner les colonnes entre train et test
x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

model = LinearRegression()
model.fit(x_train, y_train)

# # Prédiction
y_pred = model.predict(x_test)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice" : y_pred
})

submission.to_csv("submission.csv", index= False)

