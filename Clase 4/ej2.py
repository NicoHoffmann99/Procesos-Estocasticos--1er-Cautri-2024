import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

n=100
N=1000


V=np.zeros((N,n))

for i in range(n):
  V[:,i]+=np.random.normal(0,0.02,N)

X=np.zeros((N,n))

for i in range(n):
    X[:,i]=np.convolve(np.ones(10),V[:,i],'same')

mu=np.zeros(n)
var=np.zeros(n)
for i in range(n):
    mu[i]=np.mean(X[:,i])
    var[i]=np.var(X[:,i])

samp=np.linspace(0,100,100)

for i in range(n):
    plt.plot(samp,X[i])

plt.plot(samp,mu)
plt.show()

plt.plot(samp,var)
plt.show()

'''

'''
