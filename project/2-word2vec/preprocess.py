#coding=utf-8
import numpy as np
class preprocess():
    # 主要功能：去除缺失值
    def removezero(self, x, y):
        nozero = np.nonzero(y)
        y = y[nozero]
        x = np.array(x)
        x = x[nozero]
        return x, y
