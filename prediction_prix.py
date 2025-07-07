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

model = LinearRegression()
model.fit(x_train, y_train)

