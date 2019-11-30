import pickle as pk
import numpy as np
from time import time

dataset = 'gowalla'
data_path = '../Data/' + dataset + '/train_temp/temp_200_1024/'
for i in range(200):
    pkl_name = 'epoch_' + str(i) + '.pkl'
    print(pkl_name)
    pkl = pk.load(open(data_path + pkl_name, 'rb'))
    narr = np.array(pkl, dtype=np.int32)
    np.save(data_path + 'epoch_' + str(i) + '.npy', narr)
# t2 = time()
# c = np.load('b.npy')
# t3 = time()
# print(t3 - t2)

'''
t0 = time()
a = pk.load(open('Data/gowalla/train_temp/temp_200_1024/epoch_1.pkl', 'rb'))
t1 = time()
b = np.array(a)
np.save('b.npy', b)
t2 = time()
c = np.load('b.npy')
t3 = time()
print(len(a), len(a[0]), len(a[0][0]), t1 - t0, b.shape, t3 - t2)
'''