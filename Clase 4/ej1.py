import numpy as np
import matplotlib.pyplot as plt

n=100
N=1000
p=0.7

X=np.zeros((N,n))

for i in range(n):
  X[:,i]+=2*np.random.binomial(1,p,N)-1

mu=np.zeros(n)
var=np.zeros(n)
for i in range(n):
    mu[i]+=np.mean(X[:,i])
    var[i]+=np.var(X[:,i])

samp=np.linspace(0,100,100)

for i in range(N):
    plt.plot(samp,X[i])

plt.plot(samp,mu)
plt.show()

plt.plot(samp,var)
plt.show()
'''
X(n)=2P-1
E[X]=2E[P]-1=2p-1
var(X)=E[(X-u)²]=...E[4Z²]
'''
