# Deployment Template (Flask + FastAPI)

## Flask Version
Run:
```
python app_flask.py
```
Access in browser: http://127.0.0.1:5000

## FastAPI Version
Run:
```
uvicorn app_fastapi:app --reload
```
Access documentation: http://127.0.0.1:8000/docs

Both versions assume:
- trained model in `artifacts/model_randomforest.joblib`
- scaler in `artifacts/scaler.joblib`
