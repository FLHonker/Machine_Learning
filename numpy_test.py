from numpy import *
randmat = mat(random.rand(4, 4))
randmat_ = randmat.I #求逆矩阵
e = randmat*randmat_
print(e) #存在误差
print(e-eye(4))