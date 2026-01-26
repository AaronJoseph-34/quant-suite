# quant-suite
Quant Suite is a full-stack quantitative finance platform for market data ingestion, machine-learning price forecasting, options pricing, and portfolio risk analytics, built with FastAPI, Python ML models, and a modern Next.js dashboard.

# Quant Suite

A full-stack quantitative finance platform featuring:

- 📈 Real market data ingestion (Alpha Vantage + Yahoo fallback)
- 🤖 Machine learning price forecasting (ARIMA + ensemble ML)
- 🧮 Options pricing (Black-Scholes, implied volatility, Monte Carlo)
- 📊 Interactive dashboards (Next.js + Recharts)
- ⚙️ FastAPI backend with production-ready APIs

## Tech Stack

### Frontend
- Next.js
- TypeScript
- Recharts

### Backend
- FastAPI
- NumPy, SciPy
- scikit-learn
- XGBoost
- LightGBM

## Local Development

### Backend
```bash
cd backend
uvicorn app.main:app --reload

## Frontend
cd frontend
npm run dev