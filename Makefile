run_tests:
	python3 -m unittest

format_test_files:
	@./scripts/format_test_files.sh


run_preprocessing:
	python3 -m preprocessing

run_model:
	python3 -m model
