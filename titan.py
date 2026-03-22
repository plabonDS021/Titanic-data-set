
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

PATH = "/kaggle/input/competitions/titanic/" if os.path.exists("/kaggle") else "./data/titanic/"
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

# ### Obsevations 
# -women survived at a much higher rate than men (-74% vs 19%)
# - 1st class passengers survived more than 3rd c;ass (63% vs 24%)
# - most passengers were age between 20 and 40
# - data is slightly imbalanced - more deaths than survivals

## Feature Engineering

def engineer_features(df):
    df=df.copy()
    df['Age']= df['Age'].fillna(df['Age'].median())
    df['Embarked']= df['Embarked'].fillna('S')
    df['Fare']= df['Fare'].fillna(df['Fare'].median())
    df['FamilySize']=df['SibSp'] + df['Parch'] +1
    df['Sex']=df['Sex'].map({'male':0, 'female':1})
    df['Embarked']= df['Embarked'].map({'S':0, 'C':1,'Q':2})
     #extract title from name
    df['Title']=df['Name'].str.extract(r' ([A-Za-z]+)\.')

    #group rare titles together
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countries', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkher', 'Dona'],
        'Rare'
    )

    #fix some inconsistent titles
    df['Title'] = df['Title'].replace('Miles', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    ## encode title to number
    df['Title'] = df['Title'].map({
        'MR':0, 'Miss':1, 'Mrs':2, 'Master':3, "Rare":4
    })

    #fill any remaining nulls
    df['Title'] = df['Title'].fillna(0)

    ##is the passenger is travelling alone
    df['IsAlone'] = (df['FamilySize'] ==1).astype(int)

    return df


train = engineer_features(train)
test = engineer_features(test)

## Train the Model
FEATURES =['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title', 'IsAlone']
X=train[FEATURES]
y=train['Survived']
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy: {cv_score.mean(): .4f} ± {cv_score.std(): .4f}')

##Evaluation

model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=FEATURES)
importances.sort_values().plot(kind='barh', title='Feature importance')
plt.tight_layout()
plt.show()

y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred) 
disp = ConfusionMatrixDisplay(cm, display_labels=['Died', 'Survived'])
disp.plot()
plt.title('Confusion matrix')
plt.show()

## Submission File
preds = model.predict(test[FEATURES])

submission =pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': preds
})

submission.to_csv('Submission.csv', index=False)
print('Done!')
submission.head()