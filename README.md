Per eseguire il test con codice originale:
	python selfdeblur_originale.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ 
	
Per eseguire il test con variazione totale:
	python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --tv True 
	
Per eseguire il test con canny:
	python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --canny True 
	
Per eseguire il test con canny + variazione totale:
	python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --tv True --canny True
	
Per eseguire il test con rumore e varianza=0.001:
	python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --noise 0.001
	
Per eseguire il test con rumore e varianza=0.001 e salvare l'immagine con rumore:
	python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --noise 0.001 --save_noise True
