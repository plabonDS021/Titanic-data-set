
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

## Exploratory Data Analysis

print('Missing value in Train:')
print(train.isnull().sum())
print('-------')
print('Missing value in test:')
print(test.isnull().sum())

### EDA charts

fig,axes = plt.subplots(2,2, figsize=(14, 10))

train['Survived'].value_counts().plot(kind='bar',ax=axes[0,0])
axes[0,0].set_title('Survival Count')

sns.barplot(data=train, x ='Sex' , y= 'Survived', ax=axes[0,1])
axes[0,1].set_title('Surviaval by sex')

sns.barplot(data=train, x='Pclass', y='Survived', ax=axes[1,0])
axes[1,0].set_title('Survival by class')

sns.histplot(train['Age'].dropna(), bins=30, ax=axes[1,1])
axes[1,1].set_title('Age Distribution')

plt.tight_layout()
plt.show()