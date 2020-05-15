from django.shortcuts import render
from django.http import HttpResponse

# ML imports
import pandas as pd
import joblib
import keras
import numpy as np


# Create your views here.


def index(request):
    context = {'a': 'Hello World'}
    return render(request, 'app/index.html', context=context)


# Main Predict function


# Load the model, the data pipeline and the target scaler
loaded_model = keras.models.load_model('./ml_model/MPG_keras_NN')
loaded_pipeline = joblib.load('./ml_model/data_pipeline.pkl')
loaded_target_scaler = joblib.load('./ml_model/target_scaler.pkl')


def get_input_set(input_dict, load_pipeline):
    # prepare input data
    temp_data = pd.DataFrame({'x': input_dict}).transpose()
    temp_data = temp_data.apply(pd.to_numeric)
    temp_scaled = load_pipeline.transform(temp_data)[:, 1:]
    return temp_scaled


def get_prediction(load_model, load_target_scaler, input_feature_set):
    return float(np.squeeze(load_target_scaler.inverse_transform(load_model.predict(input_feature_set))))


def mpg_predict(request):
    context = {'a': 'Hello World!'}

    if request.method == 'POST':
        # print(request.POST.dict())
        model_input_values = {'acceleration': request.POST.get('accVal'),
                              'cylinders': request.POST.get('cylinderVal'),
                              'displacement': request.POST.get('dispVal'),
                              'horsepower': request.POST.get('hrsPwrVal'),
                              'model year': request.POST.get('modelVal'),
                              'origin': request.POST.get('originVal'),
                              'weight': request.POST.get('weightVal')}

        model_input = get_input_set(model_input_values,loaded_pipeline)
        prediction = get_prediction(loaded_model, loaded_target_scaler, model_input)
        # print(prediction)
        context = {'prediction': prediction}

    return render(request, 'app/index.html', context=context)
