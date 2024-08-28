## Ali Akbar Naqvi
## Internship Project 2
## Iris Flower Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

Iris= pd.read_csv('IRIS.csv')

sns.heatmap(Iris.isnull(),cmap='viridis', cbar=True)

sns.FacetGrid(Iris,hue="species",height=6).map(plt.scatter,"petal_length", "sepal_width").add_legend()

sns.pairplot(Iris[['sepal_length','sepal_width','petal_length','petal_width','species']], hue='species', diag_kind='kde')
plt.show()

flower_mapping={'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
Iris['species']=Iris['species'].map(flower_mapping)
##print(Iris.shape)
##print(Iris.info())

print(Iris)