from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

def forecast_view(request):
    # Load the historical sales data into a pandas DataFrame
    data = pd.read_csv("sales_data.csv")

    # Fit an ARIMA model to the sales data
    model = ARIMA(data["sales"], order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast the demand for the next 5 days
    forecast = model_fit.forecast(steps=5)[0]

    context = {
        "forecast": forecast,
    }
    return render(request, "forecast.html", context)