import numpy as np
import matplotlib.pyplot as plt

n=100
N=1000
p=0.7

X=np.zeros((N,n))

for i in range(n):
  X[:,i]+=2*np.random.binomial(1,p,N)-1

Y=np.zeros((N,n))

for j in range(N):
    for i in range(n-1):
        Y[j,i+1]=X[j,i]+Y[j,i]

mu=np.zeros(n)
var=np.zeros(n)
for i in range(n):
    mu[i]+=np.mean(Y[:,i])
    var[i]+=np.var(Y[:,i])

samp=np.linspace(0,100,100)

for i in range(N):
    plt.plot(samp,Y[i])

plt.plot(samp,mu,'k')
plt.show()

plt.plot(samp,var)
plt.show()
