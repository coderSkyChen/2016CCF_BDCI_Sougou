# coding=utf-8
import classify
import preprocess
import pandas as pd
import numpy as np
import csv
import codecs
import multiprocessing
import time


def input(trainname):
    """
    load the text file
    :param trainname: path of the input file
    :return:list
    """
    traindata = []
    with open(trainname, 'rb') as f:
        reader = csv.reader(f)
        count = 0
        for line in reader:
            try:
                traindata.append(line[0])
                count += 1
            except:
                print "error:", line, count
                traindata.append("1 ")
    return traindata
def output(filename, ID, age, gender, education):
    """
    generate the submit file
    :param filename: path of the submit file
    :param ID: user ID
    :param age:predicted age
    :param gender:predicted gender
    :param education:predicted education
    :return:submit file
    """
    print ID.shape, age.shape, gender.shape, education.shape
    with codecs.open(filename, 'w', encoding='gbk') as f:
        count=0
        for i in range(len(ID)):
            # if count>=1000:
            #     break
            f.write(str(ID[i]) + ' ' + str(age[i]) + ' ' + str(gender[i]) + ' ' + str(education[i]) + '\n')
            count+=1
if __name__ == '__main__':
    """
    the main function
    注意路径
    """
    start=time.time()
    # order='predict' #execute predict function
    order='test' #execute 2-fold validation function
    print 'orderis ', order
    print '----------start----------'

    #loading
    trainname = 'jieba_train_cut.csv'
    testname = 'jieba_test_cut.csv'
    traindata = input(trainname)
    testdata = input(testname)
    label_genderfile_path = 'train_gender.csv'
    label_agefile_path = 'train_age.csv'
    label_edufile_path = 'train_education.csv'
    genderdata = np.loadtxt(open(label_genderfile_path, 'r')).astype(int)
    agedata = np.loadtxt(open(label_agefile_path, 'r')).astype(int)
    educationdata = np.loadtxt(open(label_edufile_path, 'r')).astype(int)

    # ---------------------------------
    print '预处理中..'
    preprocessob = preprocess.preprocess()

    #remove label missed samples
    gender_traindatas, genderlabel = preprocessob.removezero(traindata, genderdata)
    age_traindatas, agelabel = preprocessob.removezero(traindata, agedata)
    edu_traindatas, edulabel = preprocessob.removezero(traindata, educationdata)

    # 填写你的wv向量路径
    w2vtrain = np.load('wv300_win100.train.npy')
    w2vtest = np.load('wv300_win100.test.npy')

    wv_gender_traindatas, genderlabel = preprocessob.removezero(w2vtrain, genderdata)
    wv_age_traindatas, agelabel = preprocessob.removezero(w2vtrain, agedata)
    wv_edu_traindatas, edulabel = preprocessob.removezero(w2vtrain, educationdata)

    if order=='test':
        termob1 = classify.term()
        termob2 = classify.term()
        termob3 = classify.term()
        p1 = multiprocessing.Process(target=termob1.validation,
                                     args=(gender_traindatas, genderlabel, wv_gender_traindatas, 'gender',))
        p2=multiprocessing.Process(target=termob2.validation,args=(age_traindatas, agelabel, wv_age_traindatas, 'age',))
        p3=multiprocessing.Process(target=termob3.validation,args=(edu_traindatas, edulabel, wv_edu_traindatas, 'edu',))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()
    elif order=='predict':
        termob = classify.term()
        gender=termob.predict(gender_traindatas, genderlabel, testdata, wv_gender_traindatas, w2vtest, 'gender')
        age=termob.predict(age_traindatas, agelabel, testdata, wv_age_traindatas, w2vtest, 'age')
        edu=termob.predict(edu_traindatas, edulabel, testdata, wv_edu_traindatas, w2vtest, 'edu')
        ID = pd.read_csv('user_tag_query.10W.TEST.csv').ID
        output('submit.csv', ID, age, gender, edu)

    end=time.time()
    print 'total time is', end-start
