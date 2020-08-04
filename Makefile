current_dir = $(shell pwd)

PROJECT = program_synthesis
VERSION ?= latest

.POSIX:
check:
	!(grep -R /tmp ./tests)
	flake8 --count program_synthesis
	pylint program_synth
	black --check program_synthesis

.PHONY: test
test:
	find -name "*.pyc" -delete
	pytest -s
