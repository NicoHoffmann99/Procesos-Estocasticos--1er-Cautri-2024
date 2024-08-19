import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft


def auto_corr(y):
    N=len(y)
    r=np.zeros(N)
    for k in range(N-1):
       for i in range(N-k):
            r[k] += y[i]*y[i+k]
    r=r/N
    return r


def Periodograma(y,N_fft):
    N=len(y)
    psd=np.power(np.absolute(fft(y,N_fft)),2)
    w=np.linspace(0,2*np.pi,N_fft)
    psd=psd/N
    return w, psd

def main():
    N=400
    w_1=0.42*np.pi
    w_2=0.4225*np.pi

    n=np.linspace(0,N,N)
    

    x_1=np.sin(w_1*n)
    x_2=np.sin(w_2*n)

    v=np.random.normal(0,0.01,N)

    x=x_1+x_2+v
    w, S_x_1=Periodograma(x_1,5000)
    w, S_x_2=Periodograma(x_2,5000)
    w, S_x=Periodograma(x,5000)
    plt.plot(w/np.pi,S_x)
    plt.plot(w/np.pi,S_x_1)
    plt.plot(w/np.pi,S_x_2)
    plt.xlim(0.4,0.47)
    plt.show()

main()
    