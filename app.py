from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)

model = joblib.load("credit_card_model.pkl")

gender_encoder = joblib.load("gender_encoder.pkl")
car_encoder = joblib.load("car_encoder.pkl")
realty_encoder = joblib.load("realty_encoder.pkl")
income_encoder = joblib.load("income_encoder.pkl")
education_encoder = joblib.load("education_encoder.pkl")
family_encoder = joblib.load("family_encoder.pkl")
housing_encoder = joblib.load("housing_encoder.pkl")
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():

    try:

        gender = request.form["gender"]
        own_car = request.form["own_car"]
        own_realty = request.form["own_realty"]

        children = int(request.form["children"])
        income = float(request.form["income"])

        income_type = request.form["income_type"]
        education = request.form["education"]
        family = request.form["family_status"]
        housing = request.form["housing_type"]

        birth = int(request.form["days_birth"])
        employed = int(request.form["days_employed"])

        mobil = int(request.form["mobil"])
        work_phone = int(request.form["work_phone"])
        phone = int(request.form["phone"])
        email = int(request.form["email"])

        family_members = float(request.form["family_members"])

        gender = gender_encoder.transform([gender])[0]
        own_car = car_encoder.transform([own_car])[0]
        own_realty = realty_encoder.transform([own_realty])[0]

        income_type = income_encoder.transform([income_type])[0]
        education = education_encoder.transform([education])[0]
        family = family_encoder.transform([family])[0]
        housing = housing_encoder.transform([housing])[0]

        features = np.array([[
            gender,
            own_car,
            own_realty,
            children,
            income,
            income_type,
            education,
            family,
            housing,
            birth,
            employed,
            mobil,
            work_phone,
            phone,
            email,
            family_members
        ]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Approved ✅"
            color = "green"
        else:
            result = "Rejected ❌"
            color = "red"

        return render_template(
            "result.html",
            prediction=result,
            color=color
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction="Error",
            color="red",
            error=str(e)
        )

if __name__ == "__main__":
    app.run(debug=True)
