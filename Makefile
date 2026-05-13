SHELL := /bin/bash

ENV_FILE ?= .env
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8764
FRONTEND_HOST ?= 127.0.0.1
FRONTEND_PORT ?= 8765
PYTHON ?= .venv/bin/python

.PHONY: run backend frontend lint verify-tools

run:
	@set -euo pipefail; \
	if [[ ! -f "$(ENV_FILE)" ]]; then \
		echo "Missing env file: $(ENV_FILE)"; \
		exit 1; \
	fi; \
	if [[ ! -x "$(PYTHON)" ]]; then \
		echo "Missing Python executable: $(PYTHON)"; \
		exit 1; \
	fi; \
	if [[ ! -d "frontend/node_modules" ]]; then \
		echo "Missing frontend/node_modules. Run: cd frontend && npm install"; \
		exit 1; \
	fi; \
	if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:$(BACKEND_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Backend port $(BACKEND_PORT) is already in use. Stop that process first, or set BACKEND_PORT."; \
		exit 1; \
	fi; \
	if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:$(FRONTEND_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Frontend port $(FRONTEND_PORT) is already in use. Stop that process first, or set FRONTEND_PORT."; \
		exit 1; \
	fi; \
	set -a; \
	source "$(ENV_FILE)"; \
	set +a; \
	backend_pid=""; \
	frontend_pid=""; \
	cleanup() { \
		if [[ -n "$$frontend_pid" ]] && kill -0 "$$frontend_pid" 2>/dev/null; then kill "$$frontend_pid" 2>/dev/null || true; fi; \
		if [[ -n "$$backend_pid" ]] && kill -0 "$$backend_pid" 2>/dev/null; then kill "$$backend_pid" 2>/dev/null || true; fi; \
	}; \
	trap cleanup EXIT INT TERM; \
	echo "Starting langCG with env: $(ENV_FILE)"; \
	echo "Backend:  http://$(BACKEND_HOST):$(BACKEND_PORT)"; \
	echo "Frontend: http://$(FRONTEND_HOST):$(FRONTEND_PORT)"; \
	$(PYTHON) -m uvicorn app.server:app --host "$(BACKEND_HOST)" --port "$(BACKEND_PORT)" & \
	backend_pid="$$!"; \
	echo "Waiting for backend health check..."; \
	for attempt in {1..60}; do \
		if ! kill -0 "$$backend_pid" 2>/dev/null; then \
			echo "Backend process stopped before it became ready."; \
			exit 1; \
		fi; \
		if curl -fsS --noproxy "*" "http://$(BACKEND_HOST):$(BACKEND_PORT)/api/health" >/dev/null 2>&1; then \
			break; \
		fi; \
		if [[ "$$attempt" == "60" ]]; then \
			echo "Backend did not become ready at http://$(BACKEND_HOST):$(BACKEND_PORT)/api/health"; \
			exit 1; \
		fi; \
		sleep 0.5; \
	done; \
	( cd frontend && VITE_BACKEND_URL="http://$(BACKEND_HOST):$(BACKEND_PORT)" npm run dev -- --host="$(FRONTEND_HOST)" --port="$(FRONTEND_PORT)" ) & \
	frontend_pid="$$!"; \
	while kill -0 "$$backend_pid" 2>/dev/null && kill -0 "$$frontend_pid" 2>/dev/null; do \
		sleep 1; \
	done; \
	echo "A langCG service stopped. Shutting down the remaining process."; \
	exit 1

backend:
	@set -a; source "$(ENV_FILE)"; set +a; $(PYTHON) -m uvicorn app.server:app --host "$(BACKEND_HOST)" --port "$(BACKEND_PORT)"

frontend:
	@set -a; source "$(ENV_FILE)"; set +a; cd frontend && VITE_BACKEND_URL="http://$(BACKEND_HOST):$(BACKEND_PORT)" npm run dev -- --host="$(FRONTEND_HOST)" --port="$(FRONTEND_PORT)"

lint:
	cd frontend && npm run lint
	$(PYTHON) -m compileall app
	$(PYTHON) scripts/verify_tool_contracts.py

verify-tools:
	$(PYTHON) scripts/verify_tool_contracts.py
