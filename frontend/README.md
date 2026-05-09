# langCG Frontend

Vite + React frontend for the local langCG agent API.

## Run Locally

From the repo root, start both the Python API and Vite frontend with:

`make run`

The command loads the repo root `.env`, starts the API at `http://127.0.0.1:8764`, and starts the frontend at `http://127.0.0.1:8765`.

Or start services manually:

1. Start the Python API from the repo root:
   `uvicorn app.server:app --host 127.0.0.1 --port 8764`
2. Install frontend dependencies:
   `npm install`
3. Start the frontend:
   `npm run dev`

The dev server proxies `/api` to `http://127.0.0.1:8764`. Set `VITE_API_BASE_URL` only when the API is hosted elsewhere.
