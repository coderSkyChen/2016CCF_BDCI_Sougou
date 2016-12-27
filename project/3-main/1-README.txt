主要功能：该部分为本次比赛的核心，包括了特征词加权、模型集成stacking和W2V的融合等等。

环境说明：
python2.7_64bit：以及必要的sklearn、numpy等工具包

文件说明：
main.py: 主调函数
preprocess.py：预处理类，主要是去除缺失值
classify.py: 实现主要功能的类文件，完成预测和交叉验证。
STFIWF.py: S-TFIWF加权的实现，被classify调用。该类基于sklearn.feature_extraction.text 我们根据提出的公式对IDF以及TF的部分进行了修改,具体在1093-1176行


