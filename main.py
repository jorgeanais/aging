from flask import Flask, render_template, request

from src import utils


# Load data and fit model
X, y = utils.get_train_data()
model = utils.fit_linear_model(X, y)


# Flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictor", methods=["POST"])
def predictor():
    x_input = float(request.form["run_input"])
    y_pred = utils.predict_birthyear(model, x_input)
    birthdate = utils.parse_decimal_year(y_pred)
    return render_template("predictor.html", y_hat=birthdate)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
