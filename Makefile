.PHONY: format lint test clean

format:
	black .
	isort .

lint:
	flake8 .
	mypy .
	black --check .
	isort --check .

test:
	pytest tests/ --cov=hvit --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

readme:
	typst compile -f svg README.typ assets/frame-{p}.svg
	typst compile -f svg --input theme=dark README.typ assets/frame-dark-{p}.svg
