import pickle
import sklearn

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
