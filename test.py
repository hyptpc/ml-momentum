# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# def lr_func(epoch):
#     lr_adam = 0.001
#     lr_init = lr_adam*100
#     return np.where( epoch < 5, lr_init, lr_init*np.exp(-(epoch-5)/3)+lr_adam )

# x = np.linspace(0, 100)
# y = lr_func(x)

# plt.plot(x, y)
# plt.hlines(0.001, 0, 100)
# plt.show()

import datetime

n = datetime.datetime.now()
print("{}{:0=2}{:0=2}-{:0=2}{:0=2}{:0=2}".format(n.year, n.month, n.day, n.hour, n.minute, n.second))

def a(x, y = None):
    print(x)
    print(y==None)

a(1)