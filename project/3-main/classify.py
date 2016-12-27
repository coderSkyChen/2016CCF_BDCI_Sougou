# coding=utf-8
import multiprocessing,Queue
from sklearn.cross_validation import KFold, StratifiedKFold
import xgboost as xgb
from STFIWF import TfidfVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression,RidgeClassifier,PassiveAggressiveClassifier,Lasso,HuberRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,gradient_boosting
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
class term(object):
    def __init__(self):
        random_rate = 8240
        clf1 = SGDClassifier(
            alpha=5e-05,
            average=False,
            class_weight='balanced',
            loss='log',
            n_iter=30,
            penalty='l2', n_jobs=-1, random_state=random_rate)
        clf2 = MultinomialNB(alpha=0.1)
        clf3 = LinearSVC(C=0.1, random_state=random_rate)
        clf4 = LogisticRegression(C=1.0,n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate)
        clf5 = BernoulliNB(alpha=0.1)
        clf6 = VotingClassifier(estimators=[('sgd', clf1),
                                            ('mb', clf2),
                                            ('bb', clf3),
                                            ('lf', clf4),
                                            ('bnb', clf5)], voting='hard')
        clf7 = SGDClassifier(
            alpha=5e-05,
            average=False,
            class_weight='balanced',
            loss='log',
            n_iter=30,
            penalty='l1', n_jobs=-1, random_state=random_rate)
        clf8 = LinearSVC(C=0.9, random_state=random_rate)
        clf9 = LogisticRegression(C=0.5, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate)
        clf10 = MultinomialNB(alpha=0.9)
        clf11 = BernoulliNB(alpha=0.9)
        clf12 = LogisticRegression(C=0.2, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate,penalty='l1')
        clf13 = LogisticRegression(C=0.8, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate,penalty='l1')
        clf14 = RidgeClassifier(alpha=8)
        clf15 = PassiveAggressiveClassifier(C=0.01, loss='squared_hinge', n_iter=20, n_jobs=-1)
        clf16 = RidgeClassifier(alpha=2)
        clf17 = PassiveAggressiveClassifier(C=0.5, loss='squared_hinge', n_iter=30, n_jobs=-1)
        clf18 = LinearSVC(C=0.5, random_state=random_rate)
        clf19 = MultinomialNB(alpha=0.5)
        clf20 = BernoulliNB(alpha=0.5)
        clf21 = Lasso(alpha=0.1, max_iter=20, random_state=random_rate)
        clf22 = Lasso(alpha=0.9, max_iter=30, random_state=random_rate)
        clf23 = PassiveAggressiveClassifier(C=0.1, loss='hinge', n_iter=30, n_jobs=-1, random_state=random_rate)
        clf24 = PassiveAggressiveClassifier(C=0.9, loss='hinge', n_iter=30, n_jobs=-1, random_state=random_rate)
        clf25 = HuberRegressor(max_iter=30)

        basemodel = [
            ['sgd', clf1],
            ['nb', clf2],
            ['lsvc1', clf3],
            ['LR1', clf4],
            ['bb',clf5],
            ['vote', clf6],
            ['sgdl1', clf7],
            ['lsvc2', clf8],
            ['LR2', clf9],
            ['nb2', clf10],
            ['bb2', clf11],
            ['LR3', clf12],
            ['LR4', clf13],
            ['rc1', clf14],
            ['pac1', clf15],
            ['rc2', clf16],
            ['pac2', clf17],
            ['lsvc3', clf18],
            ['nb3', clf19],
            ['bb3', clf20],
            ['lr5', clf21],
            ['lr6', clf22],
            ['rc3', clf23],
            ['pac3', clf24],
            ['hub', clf25],
        ]
        #####################################
        clf_svc = SVC(C=1,random_state=random_rate,cache_size=1000)

        self.base_models = basemodel
        self.LR=clf4
        self.svc = clf_svc

    def stacking(self,X,Y,T,wv_X,wv_T,kind):
        """
        ensemble model:stacking

        """
        print 'fitting..'
        models = self.base_models
        folds = list(KFold(len(Y), n_folds=5, random_state=0))
        S_train = np.zeros((X.shape[0], len(models)))
        S_test = np.zeros((T.shape[0], len(models)))

        for i, bm in enumerate(models):
            clf = bm[1]

            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = Y[train_idx]
                X_holdout = X[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        print S_train.shape,S_test.shape

        S_train = np.concatenate((S_train,wv_X),axis=1)
        S_test = np.concatenate((S_test, wv_T), axis=1)

        print S_train.shape,S_test.shape

        print 'scalering..'
        min_max_scaler = StandardScaler()
        S_train = min_max_scaler.fit_transform(S_train)
        S_test = min_max_scaler.fit_transform(S_test)
        print 'scalering over!'
        self.svc.fit(S_train, Y)
        yp= self.svc.predict(S_test)[:]
        return yp

    def validation(self, X, Y, wv_X, kind):
        """
        2-fold validation
        :param X: train text
        :param Y: train label
        :param wv_X: train wv_vec
        :param kind: age/gender/education
        :return: mean score of 2-fold validation
        """
        print '向量化中...'
        X=np.array(X)
        fold_n=2
        folds = list(StratifiedKFold(Y, n_folds=fold_n, shuffle=False,random_state=0))
        score = np.zeros(fold_n)
        for j, (train_idx, test_idx) in enumerate(folds):
            print j+1,'-fold'

            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_test = X[test_idx]
            y_test = Y[test_idx]

            wv_X_train =wv_X[train_idx]
            wv_X_test = wv_X[test_idx]

            vec = TfidfVectorizer(use_idf=True,sublinear_tf=False, max_features=50000, binary=True)
            vec.fit(X_train, y_train)
            X_train = vec.transform(X_train)
            X_test = vec.transform(X_test)

            print 'shape',X_train.shape

            ypre = self.stacking(X_train,y_train,X_test,wv_X_train,wv_X_test,kind)
            cur = sum(y_test == ypre) * 1.0 / len(ypre)
            score[j] = cur

        print score
        print score.mean(),kind
        return score.mean()

    def predict(self,X,Y,T,wv_X,wv_T,kind):
        """
        train and predict
        :param X: train text
        :param Y: train label
        :param T: test text
        :param wv_X: train wv
        :param wv_T: test wv
        :param kind: age/gender/education
        :return: array like ,predict of "kind"
        """
        print 'predicting..向量化中...'
        vec = TfidfVectorizer(use_idf=True, sublinear_tf=False, max_features=60000, binary=True)

        vec.fit(X, Y)
        X = vec.transform(X)
        T = vec.transform(T)

        print 'train size',X.shape,T.shape
        res = self.stacking(X, Y, T, wv_X, wv_T, kind)
        return res


