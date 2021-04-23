pylint:
	pylint predict_stock.py
	pylint tsprediction
	pylint datasets
	pylint figures
coverage:
	coverage run --source=tests  -m pytest
	coverage report    
