# StockSense

This repository contains the StockSense project.

## Project Layout

- `stocksense/backend` - FastAPI backend for stock analysis and predictions.
- `stocksense/frontend` - Node.js (Express + EJS) frontend dashboard.
- `stocksense/stocksense_artifacts` - pre-trained artifacts and prediction files.
- `stocksense_colab_train_fixed.ipynb` - Colab notebook for multi-company training.

## Quick Start

### Backend

```bash
cd stocksense/backend
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd stocksense/frontend
npm install
npm start
```

Frontend: http://127.0.0.1:5500
Backend API docs: http://127.0.0.1:8000/docs

## Existing Detailed Doc

A project readme also exists at `stocksense/README.md`.
