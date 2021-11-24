run_tests:
	mkdir -p ./test/model/state
	python3 -m unittest

format_test_files:
	@./scripts/format_test_files.sh


run_preprocessing:
	mkdir -p preprocessing/plots
	mkdir -p ./preprocessing/state
	python3 -m preprocessing

run_model:
	mkdir -p model/plots
	mkdir -p ./model/state
	python3 -m model

predict:
	@clang-format $(file_path) > $(file_path).formatted
	-python3 -m app $(file_path).formatted
	@rm -f $(file_path).formatted
