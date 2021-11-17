run_tests:
	mkdir -p ./test/model/state
	python3 -m unittest

format_test_files:
	@./scripts/format_test_files.sh


run_preprocessing:
	python3 -m preprocessing

run_model:
	python3 -m model

predict:
	python3 -m app $(file_path)
