from django.shortcuts import render
from django.http import HttpResponse


# ML imports
import pandas as pd
import joblib

# Create your views here.


def index(request):
    context = {'a': 'Hello World'}
    return render(request, 'app/index.html', context=context)


# Main Predict function


model = joblib.load('./ml_model/RF_ModelforMPG.pkl')


def mpg_predict(request):
    context = {'a': 'Hello World!'}

    if request.method == 'POST':
        print(request.POST.dict())
        model_input_values = {'acceleration': request.POST.get('accVal'),
                              'cylinders': request.POST.get('cylinderVal'),
                              'displacement': request.POST.get('dispVal'),
                              'horsepower': request.POST.get('hrsPwrVal'),
                              'model year': request.POST.get('modelVal'),
                              'origin': request.POST.get('originVal'),
                              'weight': request.POST.get('weightVal')}

        prediction_df = pd.DataFrame({'x': model_input_values}).transpose()
        prediction_df = prediction_df.apply(pd.to_numeric)
        print(prediction_df.dtypes)
        prediction = model.predict(prediction_df)[0]
        context = {'prediction': prediction}
        print(prediction)

    return render(request, 'app/index.html', context=context)
