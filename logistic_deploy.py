import pickle
import sklearn
import pandas as pd


class predObj:

    def predict_log(self, dict_pred):
        with open(r"sandardScalar.sav", 'rb') as f:
            scalar1 = pickle.load(f)

        with open(r"modelForPrediction.sav", 'rb') as t:
            model = pickle.load(t)

        data = [dict_pred]
        scaled_data = scalar1.fit(data).transform(data)
        predict = model.predict(scaled_data)
        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'

        return result

    def predict_log_file(self, file):
        with open(r"sandardScalar.sav", 'rb') as f:
            scalar1 = pickle.load(f)

        with open(r"modelForPrediction.sav", 'rb') as t:
            model = pickle.load(t)

        d = file
        table = []

        scaled_data = scalar1.fit(d).transform(d)
        predict = model.predict(scaled_data)

        pres = pd.DataFrame(data=predict, columns=['Type'])
        pres.replace(to_replace=[0, 1], value=['Non Diabetic', 'Diabetic'], inplace=True)
        d['Type'] = pres
        d.to_csv('predict.csv')

        table.append(['Pregnenacy', 'Gulcose', 'BP', 'Skin thickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Prediction'])
        for p, g, bp, sk, ins, bmi, dpf, age, pr in zip(d['Pregnancies'], d['Glucose'], d['BloodPressure'],
                                                        d['SkinThickness'],
                                                        d['Insulin'], d['BMI'], d['DiabetesPedigreeFunction'], d['Age'],
                                                        d['Type']):
            table.append([p, g, bp, sk, ins, bmi, dpf, age, pr])

        return table, d
