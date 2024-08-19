import numpy as np
import matplotlib.pyplot as plt


def ej_1():
    n=10000
    M=5000

    VeX=np.zeros((n,M))

    for i in range(n):
        VeX[i]=np.random.uniform(0,1,M)


    Z=np.zeros(M)

    for i in range(n):
        Z+=VeX[i]

    mu=np.mean(VeX[0])
    var=np.var(VeX[0])
    for i in range(M):
        Z[i]=(Z[i]-n*mu)/(np.sqrt(n)*var)

    plt.hist(Z,bins=30)
    plt.show()


ej_1()
