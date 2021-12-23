# Relation_Extraction

# data preprocessing
python preprocess/data_pre.py
input:"path='./开放知识提取-已修改20210824/'"
output: train and dev sequence files .txt;  train and dev relation files .json

Dataset for training after preprocess could be found in script/dataset

# training
python script/train/trainer.py  with setting in config.py

# prediction
python script/predict/predicter.py with setting in config.py

# evaluation
python script/predict/eval.py -l label.file -p predict file
