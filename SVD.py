from PIL import Image
from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt

I = Image.open('2哈.jpeg')
print(I.size)
I.show()
#彩转灰
L = I.convert('L')
print(L.size)
L.show()
#svd拆成3个矩阵
p,s,q=svd(L,full_matrices=False)
print(p.shape)
print(s.shape) #这里的S是一个浓缩特征值的一维矩阵 且倒序排列
print(q.shape)
#对中间的特征值矩阵S取K
def get_image_features(s,k):
    s_temp=np.zeros(s.shape[0])
    s_temp[:k]=s[:k] #取前K行
    s=s_temp*np.identity(s.shape[0]) #np.identity对角线为1的方阵   1维矩阵乘以单位对角矩阵变回方阵
    temp=np.dot(p,s) #乘特征向量转回图像
    temp=np.dot(temp,q)
    plt.imshow(temp)
    plt.show()

get_image_features(s,5)
get_image_features(s,50)
get_image_features(s,500)

