# Earthquake Risk Checker

This project trains a machine learning model from `dataset.csv` and runs a Streamlit website where a user can:

- search an area by name and get its coordinates
- use browser geolocation to fill current coordinates
- reverse lookup the address from latitude and longitude
- estimate local earthquake risk from historical records and recent USGS live earthquake data
- view nearby historical and recent earthquake events on a map

## Run

```powershell
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit.

## Important Safety Note

This is an educational decision-support tool, not an official earthquake warning system. Earthquakes cannot be predicted exactly by time and place. For real safety decisions, use official alerts and disaster-management guidance.

## What Was Improved

The original notebook converted latitude, longitude, depth, and magnitude to integers, losing important precision. The app keeps numeric precision, validates data, trains a reusable model, combines recent event feeds, and provides a proper website interface.

The current model uses engineered geospatial features and a Gradient Boosting regressor. It improves the test split score compared with the original notebook, but it is still limited by the dataset columns. Better real-world accuracy would require time, fault-line, plate-boundary, seismic-wave, and sensor data.
