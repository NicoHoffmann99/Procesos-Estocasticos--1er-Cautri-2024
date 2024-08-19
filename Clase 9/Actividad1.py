import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft

#Recibe señal, retorna señal mapeada 1 bit a 1 simbolo
def mapeo(x):
    y=2*x-1
    return y

#Recibe señal 
def canal_discreto(x,h):
    g=signal.lfilter(h,[1],x)
    y=g + np.random.normal(0,0.02,len(g))
    return y
'''

#                                                  x(n)
#                                                   |
#             +-------------+                       |
#             |             |                       +
#   y(n) ---> |      w      | ----> x_moño(n) --> -    ---> e(n)
#             |             |
#             +-------------+
def algoritmo_LMS(y,x,mu,M):
    y=np.insert(y,0,np.zeros(M-1))
    w=np.zeros((M,len(x)))
    e=np.zeros(len(x))
    x_moño=np.zeros(len(x))

    for i in range(M,len(y)-M):
        #Extraigo porción de y para calcular x_moño
        y_por = y[i+M:i:-1]
        x_moño[i] = np.dot(np.transpose(w[:,i]),y_por)
        e[i] = x[i] - x_moño[i]
        w[:,i+1] = w[:,i] + mu*e[i]*y_por
    
    return w, e, x_moño

def algoritmo_LMS_2(y,x,mu,M):

    w=np.zeros((M,len(x)-M))
    e=np.zeros(len(x))
    x_moño=np.zeros(len(x))

    for i in range(M,len(w[0])-1):
        #Extraigo porción de y para calcular x_moño
        y_por = y[i+M:i:-1]
        x_moño[i] = np.dot(np.transpose(w[:,i]),y_por)
        e[i] = x[i+M] - x_moño[i]
        w[:,i+1] = w[:,i] + mu*e[i]*y_por
    
    return w, e, x_moño
'''

def algoritmo_LMS(y,d,paso,M):

#                                                  d(n)
#                                                   |
#             +-------------+                       |
#             |             |                       +
#   y(n) ---> |      w      | ----> d_moño(n) --> -    ---> e(n)
#             |             |
#             +-------------+ 

    d_moño=np.zeros(len(y))
    error=np.zeros(len(y))
    w=np.zeros((M,len(y)))
    for i in range(M, len(d)-M):
        ventana = y[i:i-M:-1]
        d_moño[i-M] = np.dot(w[:,i],ventana)
        error[i-M] = d[i] - d_moño[i-M]
        w[:,i+1] = w[:,i] + paso*error[i-M]*ventana
    return w, error, d_moño

def actividad_1():

    b=np.random.binomial(1,0.5,2000)
    y=mapeo(b)
    h=[1,0.4,0.3,0.1,-0.2,0.05]
    x=canal_discreto(y,h)

    mu=0.05
    M=len(h)
    w,e,x_moño=algoritmo_LMS(x,y,mu,M)
    N=np.linspace(0,2000,len(w[0]))
    for i in range(M):
        plt.plot(N,w[i])
    plt.show()

    plt.stem(x)
    plt.xlim(0,100)
    plt.show()
    plt.stem(x_moño)
    plt.xlim(200,300)
    plt.show()
    plt.plot(N,np.power(np.abs(e),2))
    plt.show()

    m=100
    J=np.zeros(2000)
    for i in range(m):
        b=np.random.binomial(1,0.5,2000)
        y=mapeo(b)
        x=canal_discreto(y,h)
        w,e,x_moño=algoritmo_LMS(y,x,mu,M)
        J = J + np.power(np.abs(e),2)
    J=J/m
    plt.plot(N,J)
    plt.show()




actividad_1()

