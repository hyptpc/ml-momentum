import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

rng = np.random.default_rng()
x = rng.normal(505.1, 505.1*0.1/2.35, 300)
y = rng.normal(505.1, 505.1*0.1, 300)

plt.hist(x, bins = np.linspace(400, 600, 50), alpha = 0.5)
plt.hist(y, bins = np.linspace(400, 600, 50), alpha = 0.5)


plt.show()