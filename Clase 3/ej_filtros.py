import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft
import cmath

def filtro_1():

    F_1=[0,0.45,0.5,1]
    A=[1,0]
    V=[1/0.04,1/0.04]

    M=49

    H_coef=signal.remez(M,F_1,A,weight=V,fs=2)
    w, H=signal.freqz(H_coef,[1])
    print(len(H_coef))
    print(H_coef)

    plt.plot(w/np.pi,np.abs(H))
    plt.show()

    h=ifft(H)
    n=np.linspace(0,M,len(h))

    plt.plot(n,h)
    plt.show()

    z, p, k=signal.tf2zpk(H_coef,[1])
    plt.plot(z.real,z.imag,'o')
    plt.plot(p.real,p.imag,'x')
    plt.show()




def filtro_2():
    F_1=[0,0.25,0.3,0.65,0.7,1]
    A=[0,1,0]
    V=[1/0.06,1/0.1,1/0.03]

    M=45

    H_coef=signal.remez(M,F_1,A,weight=V,fs=2)
    w, H=signal.freqz(H_coef)
    plt.plot(w/np.pi,np.abs(H))
    plt.show()

    z, p, k=signal.tf2zpk(H_coef,[1,0])
    plt.plot(z.real,z.imag,'o')
    plt.plot(p.real,p.imag,'x')
    plt.show()




filtro_2()

