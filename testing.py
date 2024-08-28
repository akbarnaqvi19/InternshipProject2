## Ali Akbar Naqvi
## Internship Project 2
## Iris Flower Classification

from sklearn.datasets import load_iris
import pickle


with open('ICDT.pkl', 'rb') as file:
    model = pickle.load(file)


import warnings
warnings.filterwarnings('ignore')

Iris= load_iris()

def dt_iris_classification():
    print("Enter Values")
    sepal_length = float(input("Enter sepal length:"))
    sepal_width = float(input("Enter sepal width:"))
    petal_length = float(input("Enter petal length:"))
    petal_width = float(input("Enter petal width:"))

    users_inputs= [[sepal_length,sepal_width,petal_length,petal_width]]
    classification= model.predict(users_inputs)
    species_names= Iris.target_names[int(classification[0])]

    print(f"\n The Specie is: {species_names}")
if __name__ == "__main__":
    dt_iris_classification()