主要功能：处理原始文件，获得分词后的纯文本以及三个属性的label文件

环境要求：
python2.7-64bit
python版本的jieba分词：0.38
OS:any

注意事项：
注意文件的路径问题

三个文件
toCSV.py : 把原始数据文件转为UTF-8编码的CSV格式文件
CSVtolabel.py : 从上述CSV文件中把三个属性的label单独提取出来，做为独立的文件;并且把trainfile与testfile的文本保存为单独的文件
jieba_cut_fliter.py：调用jieba分词 完成分词与词性过滤