主要功能为基于train+test这20W个用户的搜索历史训练W2V模型，生成W2V向量并使用2-fold验证W2V性能。

环境要求：
python2.7-64bit
gensim工具包：0.13
OS:any

三个文件：
main.py：主调函数，设置不同的命令以完成不同功能
preprocess.py：预处理类，主要是去零
class_w2v.py：主功能类，完成W2V的训练、生成和验证