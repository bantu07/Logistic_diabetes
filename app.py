import json

import pandas as pd
from flask import Flask, request, render_template, send_file

from logistic_deploy import predObj

app = Flask(__name__)


@app.route("/", methods=['GET'])
def homepage():
    return render_template('home1.html')


@app.route("/predict", methods=['GET', 'POST'])
def predictRoute():
    if request.method == 'POST':
        p = request.form['preg']
        gl = request.form['gluc']
        bp = request.form['BP']
        skin = request.form['skin']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['DPF']
        age = request.form['age']

        my_dict = [p, gl, bp, skin, insulin, bmi, dpf, age]
        pred = predObj()
        res = pred.predict_log(my_dict)

        return render_template("result.html", res=res)
    else:
        return render_template("home1.html")


@app.route("/predict_bulk", methods=['GET', 'POST'])
def predict_bulk():
    if request.method == 'POST':
        f = request.files['myfile']
        f = pd.read_csv(f, index_col=False)

        pred = predObj()
        tab, d = pred.predict_log_file(f)

        return render_template('result1.html', table=json.dumps(tab))
    else:
        return render_template('home1.html')


@app.route('/csv/', methods = ['GET', "POST"])
def downloadFile():
    path = "predict.csv"
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
