## Ali Akbar Naqvi
## Internship Project 2
## Iris Flower Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pickle
import io
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings('ignore')

Iris= pd.read_csv('IRIS.csv')

flower_mapping={'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
Iris['species']=Iris['species'].map(flower_mapping)

data= Iris.values
#print(data.shape)
#print(data)
x_data=data[:,0:4]
y_data=data[:,4:5]

#print(x_data)
#print(y_data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x_data,y_data,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

model= DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

Iris=load_iris()

dot_data = io.StringIO()
export_graphviz(model, out_file=dot_data,
                feature_names=Iris['feature_names'],
                filled=True, rounded=True,
                special_characters=True,
                class_names=Iris['target_names'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


with open('ICDT.pkl','wb') as file:
    pickle.dump(model,file)