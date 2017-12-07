import numpy as np

a = np.zeros([3,1,2])
a[1,0,1] = 1
b = np.where(a == a.max())
print(a, b)
