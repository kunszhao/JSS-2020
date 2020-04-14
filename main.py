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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from warnings import filterwarnings

from experiment.IML import iml as IML
from utils import datasets
from utils.pre_data import split_trian_test_label, z_score, split_dataset
from utils.methods import *

filterwarnings('ignore')

if len(sys.argv) == 2:
    np.random.seed(int(sys.argv[1]))
    random.seed(int(sys.argv[1]))

###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

STEPS = 100

minClass = 1  # label of minority class
majClass = 0  # label of majority class

K = 3  # for KNN classifier

nbFoldValid = 5
maxNbParamTested = 100

measures_name = ['best_params', 'Accuracy', 'Recall', 'Precision', 'F1',
           'F_negative', 'G_measure', 'G_mean', 'Bal', 'MCC', 'AUC']

algorithm_names = ['IML', 'random_over_sampler', 'smote', 'adasyn',
              'smoteENN', 'smoteTomek', 'random_under_sampler', 'balanced_random_forest',
              'easy_ensemble', 'bagging', 'balanced_bagging',
                   'adaBoost', 'rusBoost', 'border_line_smote',
                    'svmsmote', 'origin_knn']

c_r_methods = ['naive_bayes', 'decision_tree', 'svm',
               'logistic_regression', 'nn', 'random_forest', 'knn', 'nn_5', 'nn_7', 'nn_9']

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################
if not os.path.exists("results"):
    try:
        os.makedirs("results")
    except:
        pass


def my_knn(k, Xtrain, Ytrain, Xtest):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xtrain, Ytrain)
    pred = clf.predict(Xtest)
    print(list(pred))
    return pred


def knn(k, Xtrain, Ytrain, Xtest):
    """
    Classic kNN function. Take as input train features and labels. And
    test features. Then compute pairwise distances between test and train.
    And for each test example, return the majority class among its kNN.
    """
    # 计算训练集和测试集所有实例之间的相互距离
    d = euclidean_distances(Xtest, Xtrain, squared=True)
    # 找出距离最近的K个邻居的标签值
    nnc = Ytrain[np.argsort(d)[..., :k].flatten()].reshape(Xtest.shape[0], k)  #
    # 找出最近的K个邻居中出现次数最多的标签值，作为预测结果
    pred = [max(nnc[i], key=Counter(nnc[i]).get) for i in range(nnc.shape[0])]
    return np.array(pred)


def knnSame(k, Xtrain, Ytrain):
    """
    A varriant of kNN. Here, we want to use the same set to learn and predict.
    We compute pairwise distances between train and itself. Then we fill the
    diagonal of this square matrix to be infinite to avoid having one
    example being among his neighbors. The class prediction is then the same
    as the classic kNN. Without adding infinite on the diagonal, we would
    obtain 0% error by looking only at the 1nearest neighbor.
    """
    d = euclidean_distances(Xtrain, squared=True)
    np.fill_diagonal(d, np.inf)
    nnc = Ytrain[np.argsort(d)[..., :k].flatten()].reshape(Xtrain.shape[0], k)
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
    if a.startswith("o"):
        # 使用上采样
        nbMinority = len(Xtrain[Ytrain == minClass])
        if nbMinority <= 5:
            sm = SMOTE(random_state=42, k_neighbors=nbMinority-1)
        else:
            sm = SMOTE(random_state=42)
        Xtrain2, Ytrain2 = sm.fit_sample(Xtrain, Ytrain)
    elif a.startswith("u"):
        # 使用下采样
        rus = RandomUnderSampler(random_state=42)
        Xtrain2, Ytrain2 = rus.fit_sample(Xtrain, Ytrain)
    else:
        # 不采样
        Xtrain2, Ytrain2 = Xtrain, Ytrain


    # 选择算法
    if algo.endswith("IML"):
        ml = IML(pClass=minClass, k=K, m=p["m"], Lambda=p["Lambda"], a=p["a"])
    # elif algo.endswith("LMNN"):
    #     ml = LMNN(k=K, mu=p["mu"], randomState=np.random.RandomState(1))
    # elif algo.endswith("GMML"):
    #     ml = GMML(t=p["t"], randomState=np.random.RandomState(1))
    # elif algo.endswith("ITML"):
    #     ml = ITML(gamma=p["gamma"], randomState=np.random.RandomState(1))

    # 训练模型
    ml.fit(Xtrain2, Ytrain2)
    Xtrain2 = ml.transform(Xtrain2)
    Xtest = ml.transform(Xtest)

    # TODO 可以添加各种分类算法
    # Apply kNN to predict classes of test examples
    # Ytest_pred = knn(K, Xtrain2, Ytrain2, Xtest)
    Ytest_pred = knn(K, Xtrain2, Ytrain2, Xtest)    # 预测结果

    # 添加对比分类器
    Ytest_pred_nb = naive_bayes(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_dt = decision_tree(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_svm = svm_svc(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_lr = lr(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_nn = nn(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_rf = random_forest(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_nn_5 = nn_5(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_nn_7 = nn_7(Xtrain2, Ytrain2, Xtest)
    Ytest_pred_nn_9 = nn_9(Xtrain2, Ytrain2, Xtest)

    # perf = {}
    # for true, pred, name in [(Ytest, Ytest_pred, "test")]:
    #     # Compute performance measures by comparing prediction with true labels
    #     tn, fp, fn, tp = confusion_matrix(true, pred,
    #                                       labels=[majClass, minClass]).ravel()
    #     perf[name] = ((int(tn), int(fp), int(fn), int(tp)))

    return Ytest_pred, Ytest_pred_nb, Ytest_pred_dt, Ytest_pred_svm, Ytest_pred_lr, \
           Ytest_pred_nn, Ytest_pred_rf, Ytest_pred_nn_5, Ytest_pred_nn_7, Ytest_pred_nn_9

###############################################################################
# define the evaluation measure


def harmonic_mean(x, y, beta=1):
    beta *= beta
    return (beta + 1) * x * y / np.array(beta * x + y)


def get_metrics(Ytest, Ytest_pred):
    """
    Compute performance measures by comparing prediction with true labels
    :param Ytest: real label
    :param Ytest_pred:  predict label
    :return:
    """
    TN, FP, FN, TP = confusion_matrix(Ytest, Ytest_pred,
                                      labels=[majClass, minClass]).ravel()
    return TN, FP, FN, TP


def Accuracy(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    accuracy = (TN + TP) / (TN + FP + FN + TP)
    return accuracy


def Precision(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    precision = TP / (TP + FP)
    return precision


def Recall(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    recall = TP / (TP + FN)
    return recall


def F1(Ytest, Ytest_pred):
    # perf = {}
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    # perf['test'] = ((int(tn), int(fp), int(fn), int(tp)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = harmonic_mean(precision, recall)
    return F1


def F_negative(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    F_negative = harmonic_mean(TN / (TN + FN), TN / (TN + FP))
    return F_negative


def G_measure(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    recall = TP / (TP + FN)
    pd = pf = FP / (FP + TN)
    G_measure = 2 * recall * (1 - pf) / (recall + (1 - pf))
    return G_measure


def G_mean(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    G_mean = np.sqrt(recall * specificity)
    return G_mean


def Bal(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    pd = recall = TP / (TP + FN)
    pf = FP / (FP + TN)
    Bal = 1 - np.sqrt(pf**2 + (1 - pd)**2) / np.sqrt(2)
    return Bal


def MCC(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    mcc = np.array([TP + FN, TP + FP, FN + TN, FP + TN]).prod()
    MCC = (TP * TN - FN * FP) / np.sqrt(mcc)
    return MCC


def get_F2(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    F2 = harmonic_mean(Precision(TP, FP), Recall(TP, FN), 2)
    return F2


def AUC(Ytest, Ytest_pred):
    auc = roc_auc_score(Ytest, Ytest_pred)
    return auc











###############################################################################
# Definition of parameters to test during the cross-validation for each algo
listParams = {}

listParams["IML"] = listP({"m": [1, 10, 100, 1000, 10000],
                           "Lambda": [0, 0.01, 0.1, 1, 10],
                           "a": np.arange(0, 1.01, 0.05)},
                          shuffle=True)

# listParams["oIML"] = listParams["IML"]
# listParams["uIML"] = listParams["IML"]

listNames = {a: [] for a in listParams.keys()}
listParametersNames = {a: {} for a in listParams.keys()}
for a in listParams.keys():
    for i, p in enumerate(listParams[a]):
        listParametersNames[a][str(p)] = p
        listNames[a].append(str(p))

r = {}  # All the results are stored in this dictionnary
datasetsDone = []

# 记录每个数据集上的每一次的最优参数组合以及对应参数组合下的十种指标
best_params_values = {name: {algo: {i: {measure: []
                                        for measure in measures_name}
                                    for i in range(STEPS)}
                             for algo in algorithm_names}
                      for name in datasets.d.keys()}

# 记录每个数据集在不同分类器下的算法的指标
clf_lr_values = {name: {clf: {i: {measure: []
                                  for measure in measures_name}
                              for i in range(STEPS)}
                        for clf in c_r_methods}
                 for name in datasets.d.keys()}

startTime = time.time()
for da in datasets.d.keys():  # For each dataset
    print(da)
    df = datasets.d[da]

    if len(sys.argv) == 2:
        np.random.seed(int(sys.argv[1]))
        random.seed(int(sys.argv[1]))

    for i in range(STEPS):
        print('第%d次循环' % i)

        # X, y = datasets.d[da][0], datasets.d[da][1]

        # 划分数据集
        X, y = split_dataset(df)
        _, X_origin_test, _, y_origin_test = split_trian_test_label(X, y)   # 保存原始的测试集
        X_train, X_valid = split_dataset(X)                 # 将训练集划分为训练集和验证集
        Xtrain, Xtest, ytrain, ytest = split_trian_test_label(X_train, X_valid)

        # 标准化
        x_train_scaled, x_test_scaled = z_score(Xtrain, Xtest)

        # 取消交叉验证
        # skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
        # foldsTrainValid = list(skf.split(Xtrain, ytrain))
        #
        # r[da] = {"valid": {a: {} for a in listParametersNames.keys()},
        #          "test": {a: {} for a in listParametersNames.keys()}}

        r[da] = {"F1": {a: {} for a in listParametersNames.keys()}}

        for a in listParametersNames.keys():  # For each algo
            nbParamToTest = len(listParametersNames[a])
            nbParamTested = 0
            for nameP in listNames[a]:  # For each set of parameters
                p = listParametersNames[a][nameP]
                # r[da]["valid"][a][nameP] = []
                # Compute performance on each validation fold
                # for iFoldVal in range(nbFoldValid):
                #     fTrain, fValid = foldsTrainValid[iFoldVal]
                #     perf = applyAlgo(a, p,
                #                      Xtrain[fTrain], ytrain[fTrain],
                #                      Xtrain[fValid], ytrain[fValid])
                #     r[da]["valid"][a][nameP].append(perf)

                nbParamTested += 1

                # 算法的预测值
                Ytest_pred, _, _, _, _, _, _, _, _, _ = \
                    applyAlgo(a, p, x_train_scaled, ytrain, x_test_scaled, ytest)

                # 计算F1值
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
        # best_params_values[da]['IML'][i]['F1'] = str(F1_value)

        print('The best params is: ' + str(best_param) + '\n The best F1 is: ' + str(best_F1))

        # 在最优参数组合下计算预测值
        # 这时候需要用到原始的1/4训练集和1/2的测试集
        params = ast.literal_eval(str(best_param))
        x_origin_train_scaled, x_origin_test_scaled = z_score(Xtrain, X_origin_test)

        Ytest_pred, Ytest_pred_nb, Ytest_pred_dt, Ytest_pred_svm, \
        Ytest_pred_lr, Ytest_pred_nn, Ytest_pred_rf, Ytest_pred_nn_5, \
        Ytest_pred_nn_7, Ytest_pred_nn_9 = \
            applyAlgo('IML', params,
                      x_origin_train_scaled, ytrain, x_origin_test_scaled, y_origin_test)

        # 记录在不同分类器下，算法的性能
        for measure in measures_name:
            if measure == 'best_params':
                pass
            else:
                clf_lr_values[da]['naive_bayes'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_nb))
                clf_lr_values[da]['decision_tree'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_dt))
                clf_lr_values[da]['svm'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_svm))
                clf_lr_values[da]['logistic_regression'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_lr))
                clf_lr_values[da]['nn'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_nn))
                clf_lr_values[da]['random_forest'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_rf))
                clf_lr_values[da]['knn'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred))
                clf_lr_values[da]['nn_5'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_nn_5))
                clf_lr_values[da]['nn_7'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_nn_7))
                clf_lr_values[da]['nn_9'][i][measure] = \
                    str(eval(measure)(y_origin_test, Ytest_pred_nn_9))

        # 记录下在最优参数组合下，IML算法的各种指标值
        best_params_values[da]['IML'][i]['Accuracy'] = str(Accuracy(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['Recall'] = str(Recall(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['F1'] = str(F1(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['Precision'] = str(Precision(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['F_negative'] = str(F_negative(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['G_measure'] = str(G_measure(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['G_mean'] = str(G_mean(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['Bal'] = str(Bal(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['MCC'] = str(MCC(y_origin_test, Ytest_pred))
        best_params_values[da]['IML'][i]['AUC'] = str(AUC(y_origin_test, Ytest_pred))

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

    # print('The best params is: ' + str(best_param) + '\n The best F2 is: ' + str(best_F2))


# with open('result_100_times.json', 'w') as f:
#     json.dump(best_params_values, f)
#
# with open('clf_lr_values.json', 'w') as f:
#     json.dump(clf_lr_values, f)

with open('./file/algo_final.json', 'w') as f:
    json.dump(best_params_values, f)

with open('./file/clf_final.json', 'w') as f:
    json.dump(clf_lr_values, f)


