# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 23:06:29 2016

@author: Erin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 22:23:12 2016

@author: Erin
"""
import numpy as np
from sklearn import datasets
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


#read csv that I downloaded from Thinkful mirror
df = pd.read_csv('C:/Users/Erin/thinkful/Unit4Lesson4/iris_data.csv', index_col=0)
iris = datasets.load_iris()
X = iris.data[:,:2] #first two features
Y = iris.target

#scatterplot of sepal length vs. width by species
plt.scatter(X[:,0], X[:,1], c=Y) 


#Perform PCA on Iris dataset
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std) 
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
        plt.scatter(Y_sklearn[Y==lab, 0],
                    Y_sklearn[Y==lab, 1],
                    label=lab,
                    c=col)
        plt.axis([-4, 4, -4, 3])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
plt.show()

#kNN of reduced dataset
#generate random point using elements in enumerate
sepallength = []
for el in enumerate(X_std):
    sepallength.append(el[1][0])
    
sepalwidth = []
for el in enumerate(X_std):
    sepalwidth.append(el[1][1])
    
random = []
random.append(np.random.uniform(min(sepallength), max(sepallength)))
random.append(np.random.uniform(min(sepalwidth), max(sepalwidth)))
print random

#calculate distance from to new point to all points
#use Euclidean distance because the measurements are numeric and in the same units
#euclidean distance is the square root of the sum of the squared differences
distances = []
count = 0
for el in X_std:
    euclid = np.sqrt((el[0]-random[0])**2 + (el[1]-random[1])**2 )
    distances.append([euclid, Y_sklearn[count]])
    count = count + 1


#sort each point by its distance from the new point and subset the 10 nearest points (so, k=10)
top10 = distances[:10]
print top10
#plot kNN
X_std = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std) 
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
        plt.scatter(Y_sklearn[Y==lab, 0],
                    Y_sklearn[Y==lab, 1],
                    label=lab,
                    c=col)
        plt.axis([-4, 4, -4, 3])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, Y)
def plot_scikit_lda(X, title):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0][Y == label],
                    y=X[:,1][Y == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_scikit_lda(X_lda_sklearn, title='Default LDA via scikit-learn')