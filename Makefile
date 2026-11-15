.PHONY: test smoke figures

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m dpmeta.portfolio_benchmark --help >/dev/null

RUN ?= results/portfolio-250

figures:
	$(PYTHON) -m dpmeta.figures --run $(RUN) --out figures
