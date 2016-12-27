# coding=utf-8
"""
add变量表示了原始文件的路径，TRAIN/TEST
csvfile表示了生成文件的信息
主要功能：把原始文件转为UTF-8格式
注意路径
"""
import csv

add = 'user_tag_query.10W.TRAIN' #path of the original train file

csvfile = file(add + '.csv', 'wb')# the path of the generated train file
writer = csv.writer(csvfile)
writer.writerow(['ID', 'age', 'Gender', 'Education', 'QueryList'])
with open(add, 'r') as f:
    for line in f:
        line.strip()
        data = line.split("\t")
        writedata = [data[0], data[1], data[2], data[3]]
        querystr = ''
        data[-1]=data[-1][:-1]
        for d in data[4:]:
           try:
                querystr += d.decode('GB18030').encode('utf8') + '\t'
           except:
               print data[0],querystr
        querystr = querystr[:-1]
        writedata.append(querystr)
        writer.writerow(writedata)

add = 'user_tag_query.10W.TEST'#path of the original test file

csvfile = file(add + '.csv', 'wb')# the path of the generated test file
writer = csv.writer(csvfile)
writer.writerow(['ID', 'QueryList'])
with open(add, 'r') as f:
    for line in f:
        data = line.split("\t")
        writedata = [data[0]]
        querystr = ''
        data[-1]=data[-1][:-1]
        for d in data[1:]:
           try:
                querystr += d.decode('GB18030').encode('utf8') + '\t'
           except:
               print data[0],querystr
        querystr = querystr[:-1]
        writedata.append(querystr)
        writer.writerow(writedata)

