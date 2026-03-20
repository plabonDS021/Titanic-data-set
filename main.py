import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


## load the data set

df= pd.read_csv(r'C:\Users\PRADIPTA\Desktop\project\titanic data set\data\titanic\train.csv')
df.head()
df.describe()
df.info()
