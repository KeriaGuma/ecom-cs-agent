PYTHON ?= .venv/bin/python

.PHONY: demo demo-local test sandbox-smoke

demo:
	docker image inspect ecom-cs-sandbox:local >/dev/null 2>&1 || docker compose build sandbox
	docker compose up --no-build --detach --wait

demo-local:
	$(PYTHON) -m uvicorn --factory cs_agent.demo:create_demo_app --host 127.0.0.1 --port 8000

test:
	$(PYTHON) -m pytest -q

sandbox-smoke:
	$(PYTHON) -m bench.sandbox_smoke --compose
