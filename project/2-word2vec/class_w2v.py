# coding=utf-8
from sklearn.cross_validation import KFold, StratifiedKFold
from gensim.models import word2vec
import xgboost as xgb
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler
class w2v():
    def __init__(self,size=300):
        random_rate = 8240
        self.size=size
        self.svc= SVC(C=1, random_state=random_rate)
        self.LR = LogisticRegression(C=1.0, max_iter=100, class_weight='balanced', random_state=random_rate, n_jobs=-1)
        self.clf = LinearSVC(random_state=random_rate)

    def fit(self, X, Y, T):
        """
        train and predict
        """
        print 'fitting..'
        self.LR.fit(X, Y)
        res = self.LR.predict(T)
        return res

    def validation(self,X,Y,kind):
        """

        使用2-fold进行验证
        """
        print 'validating...'
        fold_n=2
        folds = list(StratifiedKFold(Y, n_folds=fold_n, random_state=0))
        score=np.zeros(fold_n)
        for j, (train_idx, test_idx) in enumerate(folds):
            print j + 1, '-fold'
            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_test = X[test_idx]
            y_test = Y[test_idx]

            res = self.fit(X_train, y_train, X_test)
            cur = sum(y_test == res) * 1.0 / len(res)
            score[j] = cur
        print score, score.mean()
        return score.mean()

    def train_w2v(self, filename):
        """
        训练wv模型
        :param filename:path
        :return:none
        """
        sentences = word2vec.LineSentence(filename)  # 加载语料，要求语料为“一行一文本”的格式
        print '正在训练w2v 针对语料：',str(filename)
        print 'size is: ',self.size
        model = word2vec.Word2Vec(sentences, size=self.size, window=100,workers=48)  # 训练模型; 注意参数window 对结果有影响 一般5-100
        savepath = '20w_size_win100_' + str(self.size)+'.model' # 保存model的路径
        print '训练完毕，已保存: ', savepath,
        model.save(savepath)
    def load_trainsform(self,X):
        """
        载入模型，并且生成wv向量
        :param X:读入的文档，list
        :return:np.array
        """
        print '载入模型中'
        model = word2vec.Word2Vec.load('20w_size_win100_300.model') #填写你的路径
        print '加载成功'
        res=np.zeros((len(X),self.size))
        print '生成w2v向量中..'
        for i,line in enumerate(X):
            line=line.decode('utf-8')
            terms=line.split()
            count=0
            for j,term in enumerate(terms):
                try:#---try失败说明X中有单词不在model中，训练的时候model的模型是min_count的 忽略了一部分单词
                    count += 1
                    res[i]+=np.array(model[term])
                except:
                    1 == 1
            if count!=0:
                res[i]=res[i]/float(count) # 求均值
        return res

