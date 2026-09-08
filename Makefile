SHELL := /bin/bash

.PHONY: setup play arena zip gate profile_nps profile

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
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

profile_nps:
	uv run python -m src.scripts.profile_nps $(if $(AGENT),--agent "$(AGENT)") $(if $(DEPTH),--depth "$(DEPTH)")

.PHONY: profile-codebase

profile:
	uv run python -m src.scripts.profile_funcs $(if $(DEPTH),--depth "$(DEPTH)") $(if $(TOP),--top "$(TOP)")
