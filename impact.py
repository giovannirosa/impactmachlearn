#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pylab as pl
import time


def transfomData(lines):
    transfLines = []
    transfCol = []
    for line in lines:
        data = line.split(' ')
        values = list(map(lambda d: float(d.split(':')[1]), data[1:133]))
        transfCol.append(data[0])
        transfLines.append(values)
    return transfLines, transfCol


def loadDatasets(trainFile, testFile):
    with open(trainFile, 'r') as f1:
        trainingSet_X, trainingSet_Y = transfomData(f1.readlines())
    with open(testFile, 'r') as f2:
        testSet_X, testSet_Y = transfomData(f2.readlines())
    return trainingSet_X, trainingSet_Y, testSet_X, testSet_Y


def main(trainfile, testfile):

    # loads data
    print("Loading data...")
    trainingSet_X, trainingSet_Y, testSet_X, testSet_Y = loadDatasets(trainfile, testfile)

    # cria um kNN
    neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

    for i in range(1000, len(trainingSet_X), 1000):
        start_time = time.time()
        print('Fitting knn with ' + str(i))
        neigh.fit(trainingSet_X[:i], trainingSet_Y[:][:i])

        # predicao do classificador
        print('Predicting...')
        pred_Y = neigh.predict(testSet_X)

        # mostra o resultado do classificador na base de teste
        print(neigh.score(testSet_X, testSet_Y))

        # cria a matriz de confusao
        cm = confusion_matrix(testSet_Y, pred_Y)
        print(cm)
        print("--- %s seconds ---" % (time.time() - start_time))

    # pl.matshow(cm)
    # pl.colorbar()
    # pl.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: knn.py <trainfile> <testfile>")

    main(sys.argv[1], sys.argv[2])
    
