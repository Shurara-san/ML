# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import dill
import pandas as pd
import os

dill._dill._reverse_typemap['ClassType'] = type

# import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)


modelpath = "/app/app/models/catboost_pipeline.dill"
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to wine quality prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        description, company_profile, benefits = "", "", ""
        request_json = flask.request.get_json()
        if request_json["fixed acidity"]:
            fixed_acidity = request_json['fixed acidity']

        if request_json["volatile acidity"]:
            volatile_acidity = request_json['volatile acidity']

        if request_json["citric acid"]:
            citric_acid = request_json['citric acid']

        if request_json["residual sugar"]:
            residual_sugar = request_json['residual sugar']

        if request_json["chlorides"]:
            chlorides = request_json['chlorides']

        if request_json["free sulfur dioxide"]:
            free_sulfur_dioxide = request_json['free sulfur dioxide']

        if request_json["total sulfur dioxide"]:
            total_sulfur_dioxide = request_json['total sulfur dioxide']

        if request_json["density"]:
            density = request_json['density']

        if request_json["pH"]:
            pH = request_json['pH']

        if request_json["sulphates"]:
            sulphates = request_json['sulphates']

        if request_json["alcohol"]:
            alcohol = request_json['alcohol']

        logger.info(
            f'{dt} Data: fixed acidity={fixed_acidity}, volatile acidity={volatile_acidity}, citric acid={citric_acid}, residual sugar={residual_sugar}, '
            f' chlorides={chlorides}, free sulfur dioxide={free_sulfur_dioxide}, total sulfur dioxide={total_sulfur_dioxide}, density={density}, pH={pH},'
            f' sulphates={sulphates}, alcohol={alcohol}')
        try:
            preds = model.predict_proba(pd.DataFrame({"fixed acidity": [fixed_acidity],
                                                      "volatile acidity": [volatile_acidity],
                                                      "citric acid": [citric_acid],
                                                      "residual sugar": [residual_sugar],
                                                      "chlorides": [chlorides],
                                                      "free sulfur dioxide": [free_sulfur_dioxide],
                                                      "total sulfur dioxide": [total_sulfur_dioxide],
                                                      "density": [density],
                                                      "pH": [pH],
                                                      "sulphates": [sulphates],
                                                      "alcohol": [alcohol]}))
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = preds[:, 1][0]
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
