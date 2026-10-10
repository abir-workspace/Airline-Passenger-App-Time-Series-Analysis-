# Air Passenger Forecast App

A Streamlit app that forecasts monthly Air Passengers data with the ARIMA(2, 1, 2) model from the accompanying time-series assignment. It includes the 1949-1960 sample dataset, accepts compatible CSV uploads, and lets you download forecasts.

## Run locally on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local app URL, usually http://localhost:8501.

## CSV format

Uploaded files need these columns:

```csv
Month,Passengers
1949-01,112
1949-02,118
```

Use one observation per month with no missing months, and at least 24 observations.

## Deploy with Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, choose **Create app** and connect that repository.
3. Set the app file path to `app.py` (or `air-passenger-app/app.py` if the repository contains the parent folder).
4. Deploy. Keep `airline-passengers.csv` and `requirements.txt` in the same folder as `app.py`.
