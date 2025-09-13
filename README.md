# Image Classifier (Flask API + React + Docker)

## Quick Start (Docker)
```bash
docker compose build
docker compose up
```

* API: http://localhost:8000/healthz  
* Frontend: http://localhost:5173  

*The frontend proxies `/api/*` to `api:8000` through Nginx, so no CORS configuration is required when using Docker Compose.*  

## Local Development (no Docker)

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py   # runs on :8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev   # runs on :5173
```

*If you’re not using the Nginx proxy, update the fetch URL in `src/App.jsx` to:*  
```js
"http://localhost:8000/predict"
```
