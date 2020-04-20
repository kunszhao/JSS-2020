#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import random
import time
import os
import sys
import gzip
import json
import ast

from warnings import filterwarnings

from experiment.IML import iml as IML
from utils import datasets
from utils.pre_data import split_trian_test_label, z_score, split_dataset
from utils.methods import *
from utils.measures import *

filterwarnings('ignore')

if len(sys.argv) == 2:
    np.random.seed(int(sys.argv[1]))
    random.seed(int(sys.argv[1]))

###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

STEPS = 100


K = 3  # for KNN classifier

maxNbParamTested = 100

measures_name = ['best_params', 'Accuracy', 'Recall', 'Precision', 'F1',
                 'F_negative', 'MCC']

algorithm_names = ['IML', 'random_over_sampler', 'random_under_sampler',
                   'smote', 'smoteENN', 'smoteTomek',
                   'border_line_smote', 'svmsmote', 'adasyn',
                   'balanced_random_forest', 'easy_ensemble',
                   'bagging', 'balanced_bagging', 'rusBoost', 'origin_knn']

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################
if not os.path.exists("results"):
    try:
        os.makedirs("results")
    except:
        pass


def knn(k, Xtrain, Ytrain, Xtest):
    d = euclidean_distances(Xtest, Xtrain, squared=True)
    nnc = Ytrain[np.argsort(d)[..., :k].flatten()].reshape(Xtest.shape[0], k)  #
    pred = [max(nnc[i], key=Counter(nnc[i]).get) for i in range(nnc.shape[0])]
    return np.array(pred)


def listP(dic, shuffle=False):
    """
    Input: dictionnary with parameterName: array parameterRange
    Output: list of dictionnary with parameterName: parameterValue
    """
    # Recover the list of parameter names.
    params = list(dic.keys())
    # Initialy, the list of parameter to use is the list of values of
    # the first parameter.
    listParam = [{params[0]: value} for value in dic[params[0]]]
    # For each parameter p after the first, the "listParam" contains a
    # number x of dictionnary. p can take y possible values.
    # For each value of p, create x parameter by adding the value of p in the
    # dictionnary. After processing parameter p, our "listParam" is of size x*y
    for i in range(1, len(params)):
        newListParam = []
        currentParamName = params[i]
        currentParamRange = dic[currentParamName]
        for previousParam in listParam:
            for value in currentParamRange:
                newParam = previousParam.copy()
                newParam[currentParamName] = value
                newListParam.append(newParam)
        listParam = newListParam.copy()
    if shuffle:
        random.shuffle(listParam)
    return listParam


def applyAlgo(algo, p, Xtrain, Ytrain, Xtest, Ytest):
    """
    call the algorithm
    :param algo: algorithm name
    :param p: the parameters
    :param Xtrain: train data
    :param Ytrain: train label
    :param Xtest: test data
    :param Ytest: test label
    :return:
    """
    # non sampling
    Xtrain2, Ytrain2 = Xtrain, Ytrain

    # apply the algorithm
    if algo.endswith("IML"):
        ml = IML(pClass=minClass, k=K, m=p["m"], Lambda=p["Lambda"], a=p["a"])

    # train the model
    ml.fit(Xtrain2, Ytrain2)
    Xtrain2 = ml.transform(Xtrain2)
    Xtest = ml.transform(Xtest)

    # Apply kNN to predict classes of test examples
    Ytest_pred = knn(K, Xtrain2, Ytrain2, Xtest)    # prediction results

    return Ytest_pred


###############################################################################
listParams = {}

listParams["IML"] = listP({"m": [1, 10, 100, 1000, 10000],
                           "Lambda": [0, 0.01, 0.1, 1, 10],
                           "a": np.arange(0, 1.01, 0.05)},
                          shuffle=True)

listNames = {a: [] for a in listParams.keys()}
listParametersNames = {a: {} for a in listParams.keys()}
for a in listParams.keys():
    for i, p in enumerate(listParams[a]):
        listParametersNames[a][str(p)] = p
        listNames[a].append(str(p))

r = {}  # All the results are stored in this dictionnary
datasetsDone = []

best_params_values = {name: {algo: {i: {measure: []
                                        for measure in measures_name}
                                    for i in range(STEPS)}
                             for algo in algorithm_names}
                      for name in datasets.d.keys()}

startTime = time.time()
for da in datasets.d.keys():  # For each dataset
    print(da)
    df = datasets.d[da]

    if len(sys.argv) == 2:
        np.random.seed(int(sys.argv[1]))
        random.seed(int(sys.argv[1]))

    for i in range(STEPS):
        print('The %d times loop' % i)

        # split dataset
        X, y = split_dataset(df)
        _, X_origin_test, _, y_origin_test = split_trian_test_label(X, y)   # save original test
        X_train, X_valid = split_dataset(X)            # train dataset and validation dataset
        Xtrain, Xtest, ytrain, ytest = split_trian_test_label(X_train, X_valid)

        # normalization
        x_train_scaled, x_test_scaled = z_score(Xtrain, Xtest)

        r[da] = {"F1": {a: {} for a in listParametersNames.keys()}}

        for a in listParametersNames.keys():  # For each algo
            nbParamToTest = len(listParametersNames[a])
            nbParamTested = 0
            for nameP in listNames[a]:  # For each set of parameters
                p = listParametersNames[a][nameP]

                nbParamTested += 1

                # the prediction value for algorithm
                Ytest_pred = applyAlgo(a, p, x_train_scaled, ytrain, x_test_scaled, ytest)

                # calculate F1
                F1_value = F1(ytest, Ytest_pred)
                #
                r[da]['F1'][a][nameP] = F1_value

                print(da, a,
                      str(nbParamTested)+"/"+str(nbParamToTest),
                      "time: {:8.2f} sec".format(time.time()-startTime),
                      "test F1 {:5.2f}".format(F1_value*100), p)
                if nbParamTested >= maxNbParamTested:
                    break

        best_param = max(r[da]['F1'][a].keys(), key=(lambda x: r[da]['F1'][a][x]))
        best_F1 = r[da]['F1'][a][str(best_param)]

        best_params_values[da]['IML'][i]['best_params'] = best_param

        print('The best params is: ' + str(best_param) + '\n The best F1 is: ' + str(best_F1))

        # Calculate the predicted value under the optimal parameter combination
        # The original 1/4 training set and 1/2 test set are needed
        params = ast.literal_eval(str(best_param))
        x_origin_train_scaled, x_origin_test_scaled = z_score(Xtrain, X_origin_test)

        Ytest_pred = applyAlgo('IML', params, x_origin_train_scaled, ytrain, x_origin_test_scaled, y_origin_test)

        # Record the indicator values of the algorithm under the optimal parameter combination
        best_params_values[da]['IML'][i]['Accuracy'] = str(Accuracy(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['Recall'] = str(Recall(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['F1'] = str(F1(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['Precision'] = str(Precision(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['F_negative'] = str(F_negative(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['MCC'] = str(MCC(y_origin_test, Ytest_pred))

        for algo_name in algorithm_names:
            if algo_name == 'IML':
                pass
            else:
                # call the algorithm to predict
                y_test_pred = eval(algo_name)(x_origin_train_scaled, ytrain, x_origin_test_scaled)
                # calculate the evaluation
                for measure in measures_name:
                    if measure == 'best_params':
                        pass
                    else:
                        best_params_values[da][algo_name][i][measure] = \
                            str(eval(measure)(y_origin_test, y_test_pred))

    datasetsDone.append(da)

    # Save the results at the end of each dataset
    if len(sys.argv) == 2:
        f = gzip.open("./results/res" + sys.argv[1] + ".pklz", "wb")
    else:
        f = gzip.open("./results/res" + str(startTime) + ".pklz", "wb")

    pickle.dump({"res": r, "algos": list(listParametersNames.keys()),
                 "datasets": datasetsDone}, f)
    f.close()


with open('./file/algo_final.json', 'w') as f:
    json.dump(best_params_values, f)



