init: # Setup pre-commit
	pip3 install -r ./requirements.txt
	pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
	pre-commit install --hook-type pre-commit --hook-type pre-push

lint: # Lint all files in this repository
	pre-commit run --all-files --show-diff-on-failure

test: # Lint all files in this repository
	pytest