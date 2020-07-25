from flask import Flask, request, render_template

from logistic_deploy import predObj

app = Flask(__name__)


class ClientApi:
    def __init__(self):
        self.predObj = predObj()


@app.route("/", methods=['GET'])
def homepage():
    return render_template('home.html')


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
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
