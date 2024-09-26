from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Загрузка модели
model_path = 'model/random_forest_model.pkl'
model_path2 = 'model/xgb_model.pkl'
rf_model = joblib.load(model_path)
rf_model2 = joblib.load(model_path2)

# Пример данных для выпадающих списков
countries = ['Albania', 'Country2', 'Country3']  
items = ['Maize', 'Wheat', 'Crop3']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prediction2 = None
    if request.method == 'POST':
        country = request.form['country']
        item = request.form['item']
        pesticides = float(request.form['pesticides'])
        avg_temp = float(request.form['avg_temp'])
        rainfall = float(request.form['rainfall'])

        # Кодирование категориальных переменных 
        country_encoded = countries.index(country)
        item_encoded = items.index(item)

        # Подготовка входных данных
        input_data = np.array([[country_encoded, item_encoded, pesticides, avg_temp, rainfall]])
        
        # Прогноз
        prediction = round((rf_model.predict(input_data)[0])/10000, 4)
        prediction2 = round((rf_model2.predict(input_data)[0])/10000, 4)

    return render_template('index.html', countries=countries, items=items, prediction=prediction , prediction2=prediction2)

if __name__ == '__main__':
    app.run(debug=True)