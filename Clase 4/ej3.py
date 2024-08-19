import numpy as np
import matplotlib.pyplot as plt


N=1000

t=np.arange(0,1.99,0.01)
n=len(t)

X=np.zeros((N,n))

A=np.random.uniform(0,1,N)
B=np.random.uniform(0,1,N)

for i in range(N):
  X[i]=A[i]*t+B[i]

mu=np.zeros(n)
var=np.zeros(n)
for i in range(n):
    mu[i]=np.mean(X[:,i])
    var[i]=np.var(X[:,i])

for i in range(n):
    plt.plot(t,X[i])

plt.plot(t,mu,'k')
plt.show()

plt.plot(t,var)
plt.show()
