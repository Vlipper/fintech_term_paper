# fintech_term_paper


## Preparations
1. Положить датасет в директорию `data` в одной родительской директории с `scripts`
2. Запустить скрипт `compress_source_data.py` - он прочитает исходный `.csv` и 
сохранит его в более быстрый формат `.npz`


## There are two different approaches:
train_wave.py - for training model on raw wave signal  
train_spect.py - for training model on spectrogram of raw signal    
train_wave_cpc.py - for training CPC model   
