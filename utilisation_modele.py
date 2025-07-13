import pandas as pd
import joblib

model = joblib.load("model_prix_immobilier")
preprocessor = joblib.load("preprocessor.pkl")
selector = joblib.load("selector.pkl")

new_data = pd.read_csv("test.csv")
X_processed = preprocessor.transform(new_data)
X_selected = selector.transform(X_processed)

y_pred = model.predict(X_selected)

submission = pd.DataFrame({
    "Id" : new_data["Id"],
    "SalePrice": y_pred
})

submission.to_csv("submission_new.csv", index = False)