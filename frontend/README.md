# langCG Frontend

Vite + React frontend for the local langCG agent API.

## Run Locally

1. Start the Python API from the repo root:
   `uvicorn app.server:app --reload --port 8000`
2. Install frontend dependencies:
   `npm install`
3. Start the frontend:
   `npm run dev`

The dev server proxies `/api` to `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` only when the API is hosted elsewhere.
