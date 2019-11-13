import numpy as np
import matplotlib.pyplot as plt

"""
This file is used to test some code. and do simple plot task.
"""

# m = [22,3,4,1,4]
# print(min(m))
#
# a = np.array([[1,2,3], [4,5,6],  [7,8,9]])
# b = np.array([[1,2,3], [4,5,6]])
# K = 2
#
# d = np.reshape(a, [-1, 1, 3])
# print(d-b)
#
# a = np.tile(a, [1, K])
# a = np.reshape(a, [3, K, -1])
#
# d = np.linalg.norm(a-b, axis=-1)
# print(a)
# print(b)
# print(d)
#
# print(np.arange(10))

xx = [2, 5, 10, 15, 20, 30, 40, 50, 80, 100]
# y_akl = [1.8710, 2.1236, 1.7014, 1.6678, 1.6831, 1.6014, 1.6451, 1.6787]
# y_cosine = [1.5406, 1.8167, 1.5049, 1.7202, 1.6542, 1.5591, 1.4961, 1.5865]

y_akl_kmedoids = [1.6940407301262734, 1.8733446148722346, 1.5708255125596333, 1.6326623475164899, 1.5799032165447888, 1.4802387572161542, 1.3617701783735359, 1.3654598930189061, 1.3460701186720563, 1.3268296141960911]
y_cosine_kmedoids = [1.6497407876233803, 1.857832042394479, 1.8394688850355385, 1.8596609458244626, 1.8623697408187476, 1.7946961254330955, 1.7564821379789, 1.736556739626105, 1.708093636650098, 1.6453076824894997]
x = np.arange(len(xx))

plt.figure(figsize=(5, 4))
plt.bar(x, y_cosine_kmedoids)
plt.xticks(x, xx)
plt.ylabel('CH-score')
plt.xlabel('cluster number k')
plt.tight_layout()
# plt.show()
plt.savefig('kmedoids_cosine.png')