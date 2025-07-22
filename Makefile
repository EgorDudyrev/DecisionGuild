PYTHON = python3

.PHONY = test


test:
	pip install -e .
	pytest --cov=decisionguild --cov-report term-missing $(test_path)