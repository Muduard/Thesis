Test original method:
	```python selfdeblur_originale.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ ```
	
Test our method with total variation loss:
	```python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --tv True ```
	
Test our method with canny input:
	```python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --canny True ```
	
Test our method with canny input + total variation loss:
	```python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --tv True --canny True```
	
Test our method with gaussian noise and 0.001 variance:
	```python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --noise 0.001```
	
Test our method with gaussian noise and 0.001 variance saving generated noisy image in ```./datasets/06_02/noisy```:
	```python selfdeblur_tesi.py --data_path ./datasets/06_02/ --save_path ./results/06_02/ --noise 0.001 --save_noise True```
