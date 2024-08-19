import numpy as np
import matplotlib.pyplot as plt


x1 = np.random.rayleigh(0.5, size=100)
x2 = np.random.rayleigh(0.5, size=100)
x3 = np.random.rayleigh(0.5, size=10000)
plt.hist(x1,10)
plt.show()
plt.hist(x2,30)
plt.show()
plt.hist(x3,30)
plt.show()
