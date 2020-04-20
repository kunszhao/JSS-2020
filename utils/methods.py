import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


K = 3


"""sample methods"""


def random_over_sampler(X, y, xtest):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)


def smote(X, y, xtest):
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)


def adasyn(X, y, xtest):
    X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)


def smoteENN(X, y, xtest):
    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)


def smoteTomek(X, y, xtest):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)


def random_under_sampler(X, y, xtest):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)


def border_line_smote(X, y, xtest):
    sm = BorderlineSMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)



def svmsmote(X, y, xtest):
    sm = SVMSMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    return call_knn(X_resampled, y_resampled, xtest)


"""ensemble methods"""


def balanced_random_forest(X, y, xtest):
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
    brf.fit(X, y)
    y_pred = brf.predict(xtest)
    return y_pred


def easy_ensemble(X, y, xtest):
    eec = EasyEnsembleClassifier(random_state=0)
    eec.fit(X, y)
    y_pred = eec.predict(xtest)
    return y_pred


def bagging(X, y, xtest):
    bc = BaggingClassifier(base_estimator=KNeighborsClassifier(
        n_neighbors=3, algorithm='ball_tree'), random_state=0)
    bc.fit(X, y)
    y_pred = bc.predict(xtest)
    return y_pred


def balanced_bagging(X, y, xtest):
    bbc = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(
        n_neighbors=3, algorithm='ball_tree'),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=0)
    bbc.fit(X, y)
    y_pred = bbc.predict(xtest)
    return y_pred


def adaBoost(X, y, xtest):
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                             n_estimators=100, random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def rusBoost(X, y, xtest):
    rus_boost = RUSBoostClassifier(base_estimator=DecisionTreeClassifier(),
        n_estimators=200, algorithm='SAMME.R', random_state=0)
    rus_boost.fit(X, y)
    y_pred = rus_boost.predict(xtest)
    return y_pred


"""common methods"""


def call_knn(X, y, xtest):
    Ytest_pred = knn(K, X, y, xtest)
    return Ytest_pred


def origin_knn(X, y, xtest):
    return call_knn(X, y, xtest)


def knn(k, Xtrain, Ytrain, Xtest):
    d = euclidean_distances(Xtest, Xtrain, squared=True)
    nnc = Ytrain[np.argsort(d)[..., :k].flatten()].reshape(Xtest.shape[0], k)  #
    pred = [max(nnc[i], key=Counter(nnc[i]).get) for i in range(nnc.shape[0])]
    return np.array(pred)
