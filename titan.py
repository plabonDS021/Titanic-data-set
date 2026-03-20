
# Titanic Survival Prediction
## predicting who survived using EDA+ Machine learning

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



## load the data set

PATH = "/kaggle/input/titanic/" if os.path.exists("/kaggle") else "./data/titanic/"
train =pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Train Head:', train.head())
