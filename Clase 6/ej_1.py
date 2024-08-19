import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft


def auto_corr(y):
    N=len(y)
    r=np.zeros(N)
    for k in range(N-1):
       for i in range(N-k):
            r[k] += y[i]*y[i+k]
    r=r/N
    return r

def PSD_gen(y):
    N=len(y)
    PSD=np.power(np.absolute(y),2)
    PSD=PSD/N
    return PSD

def ej_1():
    #1)
    L=5000

    n=np.linspace(0,L,L)

    x=np.random.normal(0,1,L)
    b=[1,0.5,0.25]
    a=[1]

    x_n=signal.lfilter(b,a,x)

    plt.plot(n,x_n)
    plt.show()
    
 
    r_x_n=auto_corr(x_n)
    plt.plot(n,r_x_n)
    plt.show()
    
    

ej_1()





