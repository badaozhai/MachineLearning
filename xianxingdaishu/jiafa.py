#线性代数 ,矩阵的加法

from numpy import array
a1 = array([[1,2,3],[2,3,4]])
print(a1)            #[1 2 3]
print(type(a1))     #<class 'numpy.ndarray'>



a2 = array([[3,4,5],[1.2,3.1,4.1]])
print(a2)
a3 = a1 + a2
print(a3)

print("end")