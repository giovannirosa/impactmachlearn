#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pylab as pl
from matplotlib import ticker
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


def runTest(method, base, sets):
    trainingSet_X, trainingSet_Y, testSet_X, testSet_Y = sets
    X_cord = []
    Y_cord = []
    Y_time = []
    last_cm = None
    for i in range(1000, len(trainingSet_X) + 1, 1000):
        start_time = time.time()
        print('%s[FIT](%d)' % (method, i))
        base.fit(trainingSet_X[:i], trainingSet_Y[:][:i])

        # predicao do classificador
        print('Predicting...')
        pred_Y = base.predict(testSet_X)
        time_pred = time.time() - start_time

        # mostra o resultado do classificador na base de teste
        score = base.score(testSet_X, testSet_Y)
        print("%s[SCORE](%d): %s" % (method, i, score))
        X_cord.append(i)
        Y_cord.append(score * 100)

        # cria a matriz de confusao
        last_cm = confusion_matrix(testSet_Y, pred_Y)
        print(last_cm)

        # mostra tempo
        print("%s[TIME](%d): %s seconds" % (method, i, time_pred))
        Y_time.append(time_pred)
    return X_cord, Y_cord, Y_time, last_cm


def confusionMatrix(method, cm):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title("Matriz de Confusao - " + method)
    fig.colorbar(cax)
    ax.set_xticklabels(['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax.set_yticklabels(['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    pl.xlabel('Predito')
    pl.ylabel('Valor Real')
    pl.savefig(method + '_cm.png', bbox_inches='tight')


def main(trainfile, testfile):

    # loads data
    print("Loading data...")
    start_time = time.time()
    sets = loadDatasets(
        trainfile, testfile)
    print("Load data time: %s seconds" % (time.time() - start_time))

    # kNN
    KNN_X_cord, KNN_Y_cord, KNN_Y_time, KNN_last_cm = runTest(
        'KNN', KNeighborsClassifier(n_neighbors=3, metric='euclidean'), sets)

    # Naive Bayes
    NB_X_cord, NB_Y_cord, NB_Y_time, NB_last_cm = runTest('NAIVEBAYES', GaussianNB(), sets)

    # LDA
    LDA_X_cord, LDA_Y_cord, LDA_Y_time, LDA_last_cm = runTest(
        'LDA', LinearDiscriminantAnalysis(solver='lsqr'), sets)

    # LR
    LR_X_cord, LR_Y_cord, LR_Y_time, LR_last_cm = runTest('LR', LogisticRegression(
        random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000), sets)

    pl.figure(1)
    pl.title("Treino x Desempenho")
    pl.xlabel("Tamanho da base de treino")
    pl.ylabel("Taxa de acerto (%)")
    pl.plot(KNN_X_cord, KNN_Y_cord)
    pl.plot(NB_X_cord, NB_Y_cord)
    pl.plot(LDA_X_cord, LDA_Y_cord)
    pl.plot(LR_X_cord, LR_Y_cord)
    pl.legend(['KNN', 'Naive Bayes', 'LDA', 'LR'], loc='best')
    pl.savefig('treino_desempenho.png', bbox_inches='tight')

    pl.figure(2)
    pl.title("Treino x Tempo")
    pl.xlabel("Tamanho da base de treino")
    pl.ylabel("Tempo (segundos)")
    pl.plot(KNN_X_cord, KNN_Y_time)
    pl.plot(NB_X_cord, NB_Y_time)
    pl.plot(LDA_X_cord, LDA_Y_time)
    pl.plot(LR_X_cord, LR_Y_time)
    pl.legend(['KNN', 'Naive Bayes', 'LDA', 'LR'], loc='best')
    pl.savefig('treino_tempo.png', bbox_inches='tight')

    pl.figure(3)
    pl.title("Treino x Tempo")
    pl.xlabel("Tamanho da base de treino")
    pl.ylabel("Tempo (segundos)")
    pl.plot(NB_X_cord, NB_Y_time)
    pl.plot(LDA_X_cord, LDA_Y_time)
    pl.plot(LR_X_cord, LR_Y_time)
    pl.legend(['Naive Bayes', 'LDA', 'LR'], loc='best')
    pl.savefig('treino_tempo_semknn.png', bbox_inches='tight')

    confusionMatrix('KNN', KNN_last_cm)
    confusionMatrix('NB', NB_last_cm)
    confusionMatrix('LDA', LDA_last_cm)
    confusionMatrix('LR', LR_last_cm)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: knn.py <trainfile> <testfile>")

    main(sys.argv[1], sys.argv[2])
