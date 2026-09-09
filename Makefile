SHELL := /bin/bash

.PHONY: setup play arena zip gate test profile_nps profile eval

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/numba $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/numba

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m unittest discover -s tests
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

test:
	uv run python -m unittest discover -s tests

profile_nps:
	uv run python -m src.scripts.profile_nps $(if $(DEPTH),--depth "$(DEPTH)") $(if $(TIME),--time "$(TIME)") $(if $(POS),--position "$(POS)") $(if $(FEN),--fen "$(FEN)") $(if $(QUIET),--quiet)

.PHONY: profile-codebase

profile:
	uv run python -m src.scripts.profile_funcs $(if $(DEPTH),--depth "$(DEPTH)") $(if $(TIME),--time "$(TIME)") $(if $(TOP),--top "$(TOP)") $(if $(SORT),--sort "$(SORT)") $(if $(POS),--position "$(POS)") $(if $(FEN),--fen "$(FEN)") $(if $(QUIET),--quiet)

eval:
	uv run python -m src.scripts.eval $(if $(GAMES),--games "$(GAMES)") $(if $(BASE_MS),--base-time "$(BASE_MS)") $(if $(INC_MS),--inc "$(INC_MS)") $(if $(SKILL),--skill "$(SKILL)") $(if $(STATIC),--static-skill) $(if $(WORKERS),--workers "$(WORKERS)") $(if $(TRACE_TIMING),--trace-timing) $(if $(POSITIVE_ROOT_GAP),--positive-root-gap) $(if $(ASPIRATION_RESERVE),--aspiration-reserve "$(ASPIRATION_RESERVE)")
