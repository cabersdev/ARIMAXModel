install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

format:
	black src/ tests/

lint:
	flake8 src/ tests/

data:
	dvc repro

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf .pytest_cache