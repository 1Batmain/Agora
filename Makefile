# Agora — raccourcis. `make setup` une fois, puis `make dev`.
UV_EXTRAS = --extra contender --extra embed-contender --extra faiss
.PHONY: setup dev back front build test

setup: ## installe tout (deps, caches, secrets)
	./scripts/setup.sh

back: ## backend FastAPI :8010 (serve-only)
	AGORA_CLAIMS_BACKEND=api uv run $(UV_EXTRAS) --with fastapi --with uvicorn \
	  uvicorn backend.server:app --host 0.0.0.0 --port 8010

front: ## front Vite :5180
	cd frontend && npm run dev

dev: ## back + front en parallèle
	$(MAKE) -j2 back front

build: ## build de prod du front
	cd frontend && npm run build

test: ## tests backend (pytest)
	AGORA_CLAIMS_BACKEND=api uv run $(UV_EXTRAS) --with fastapi --with pytest --with httpx \
	  pytest backend/tests -p no:warnings
