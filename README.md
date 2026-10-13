# Wi-FiScope — Python-backed final application

This version keeps the existing Wi-FiScope UI while moving the **actual analysis source of truth to Python**.

## Architecture

Browser UI:
`index.html`

Python web server:
`app.py` (Flask)

Mathematical analysis:
`analysis.py`

Data:
`data/wifi_data_synthetic.csv`

The browser calls:

- `POST /api/analyze` for the complete project pipeline
- `POST /api/predict` for an individual unmeasured-coordinate estimate
- `GET /api/health` for backend status

## Mathematical pipeline implemented in Python

1. Location × time RSSI matrix
2. Mean-centering + L2 normalization
3. Inner product / cosine similarity
4. Euclidean norm distance
5. Least-squares spatial model
6. Held-out validation + MAE / RMSE / R²
7. Persistent weak-zone classification
8. Grid-based spatial reference used by the UI
9. Eigenvalue/eigenvector analysis

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open:

`http://127.0.0.1:5000`

Do **not** double-click `index.html` for the final demo. Launch `app.py` so the browser interface uses the Python backend.

## Real survey data

Replace `data/wifi_data_synthetic.csv` for backend sample access if desired, and use the app's **Load Survey CSV** control for the team's final field measurements.

Required columns:

```text
Location_ID
Location_Name
Floor
X_m
Y_m
Time_Period
RSSI_dBm
```

Optional:

```text
SSID
BSSID
```

## Important academic note

The development dataset is synthetic. Replace it with actual survey readings before final review and regenerate the final screenshots/results.
