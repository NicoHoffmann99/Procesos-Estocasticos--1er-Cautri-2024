import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft

archivos=[r"C:\Users\nicoh\OneDrive\Documentos\Facultad\Procesos Estocasticos\TP2\eeg_ojos_abiertos_t7.csv",r"C:\Users\nicoh\OneDrive\Documentos\Facultad\Procesos Estocasticos\TP2\eeg_ojos_cerrados_t7.csv"]
#FUNCIONES DE USO GENERAL

#Recibe señal(arreglo) => Devuelve función de autocorrelación
def auto_corr(y):
    N=len(y)
    r=np.zeros(N)
    for k in range(N-1):
       for i in range(N-k):
            r[k] += y[i]*y[i+k]
    r=r/N
    return r

#Estimador de periodograma
#Recibe señal(arreglo) y parametro para realizar la fft de la señal
#=> Devuelve periodograma
def Periodograma(y,N_fft):
    N=len(y)
    psd=np.power(np.abs(fft(y,N_fft)),2)
    #psd=psd/N
    return psd/N

#Recibe señal(arreglo) => devuelve potencia
def potencia(x):
    p=0
    for i in range(len(x)):
        p+=np.power(np.abs(x[i]),2)
    return p/len(x)

#FUNCIONES EJERCICIO 1
#Recibe arreglo coef a, float ganancia G y arreglo dominio w
#Devuelve la PSD
def PSD_model(a,G,w):
    total_sum = 0
    for k in range(1, len(a) + 1):
        total_sum += np.multiply(a[k - 1], np.exp(1j * -w * k))
    denominator = np.power(abs(1 - total_sum), 2)
    return np.power(G, 2) / denominator

#Recibe señal(arreglo) y orden P
#Devuelve coeficiente a y ganancia G
def ar_model(x,P):
    r_x=auto_corr(x)

    #Vector r donde r=[R(k=1),R(k=2),...,R(k=P)] , no incluye R(k=0)
    r_p=r_x[1:P+1]

    #Matriz R
    R_p=np.zeros((P,P))
    for i in range(P):
        for j in range(P):
            R_p[i,j]=r_x[np.abs(j-i)]
    
    
    #coeficientes a=R^(-1) * r
    a=np.matmul(np.linalg.inv(R_p),r_p)
  
    #calculo la ganancia con G=(R(0) - sum(a_i*r_i))^(1/2)
    G=r_x[0] 
    for i in range(P):
        G-=a[i]*r_p[i]
    G=np.sqrt(G)
    print(a)
    print(G)
    return a, G

#FUNCIONES EJERCICIO 2

#Recibe señal(arreglo), porcentaje de solapamiento(pwe_margin), tipo de ventana(wind), tamaño de ventana M
def welch_metod_PSD(x,per_margin,wind,M):
    N=len(x)
    K=int(M*(1-per_margin/100)) #Desplazamiento
    L=int(N/K) #Cantidad de segmetos

    v=signal.get_window(wind,M)
    V=potencia(v) #Calculo la potencia de la ventana

    S_w=np.zeros(N)
    for i in range(L-1):  
        x_i=x[i*K : (i*K + M)]*v
        S_w+=Periodograma(x_i,N)/V
    return S_w/L

#FUNCIONES EJERCICIO 3
#Recibe arreglo coef a, float ganancia G y tamaño deseado de señal sintetica
#devuelve señal sintetica
def sintex_EEG(a,G,N):
    a = np.insert(-a,0,1) #Redefino polos agregando 1 de orden 0 y multiplicando por -1 los polos de mayor orden
    u=np.random.normal(0,1,N)
    return signal.lfilter([G],a,u)

#FUNCIONES EJERCICIO 4
#Recibe orden M, Ganancias A, Rango de Frecuencias F, Tolerancias V, f de muestro, tamaño del arreglo del filtro N
#devuelve dominio w y valor absoluto de la respuesta en frecuencia H
def FIR_filter(M,A,F,V,f,N):
    b=signal.remez(M,F,A,weight=V,fs=f)
    w, H=signal.freqz(b,[1],worN=N)
    return w, np.abs(H)

#Recibe señal(en el tiempo) y filtro(en frecuencias)
#devuelve señal filtrada(en tiempo)
def signal_filtering(x,H):
    X=fft(x)
    return ifft(X*H)


def ej_1():
    P=[2,13,30]
    Label=['Orden 2', 'Orden 13', 'Orden 30']
    Titles=['PSD - EEG ojos abiertos','PSD - EEG ojos cerrados']
    x = [np.loadtxt(archivos[0], dtype=float) ,np.loadtxt(archivos[1], dtype=float)]

    for j in range(len(x)):
        w_1=np.linspace(0,2*np.pi,len(x[j]))
        S_e=Periodograma(x[j],len(x[j]))
        plt.plot(w_1,10*np.log10(S_e),label='Periodograma')
        for i in range(len(P)):
            a, G=ar_model(x[j],P[i])
            S_t=PSD_model(a,G,w_1)
            plt.plot(w_1,10*np.log10(S_t), label=Label[i])
        plt.title(Titles[j])
        plt.ylabel('Periodograma - PSD [dB]')
        plt.xlabel('w[rad/s]')
        plt.legend()
        plt.grid()
        plt.xlim(0,np.pi)
        plt.show()

def ej_2():
    P=13
    x = [np.loadtxt(archivos[0], dtype=float) ,np.loadtxt(archivos[1], dtype=float)]
    Titles=['PSD y Método de Welch- EEG ojos abiertos','PSD y Método de Welch- EEG ojos cerrados']
    for i in range(len(x)):
        a, G=ar_model(x[i],P)
        
        S_w = welch_metod_PSD(x[i],50,'hamming',80)
        w_1=np.linspace(0,2*np.pi,len(S_w))
        S_t=PSD_model(a,G,w_1)
        plt.plot(w_1,10*np.log10(S_w),label='Método Welch')
        plt.plot(w_1,10*np.log10(S_t),label='Teórico - P=13')
        plt.title(Titles[i])
        plt.ylabel('Periodograma(Welch) - PSD [dB]')
        plt.xlabel('w[rad/s]')
        plt.legend()
        plt.grid()
        plt.xlim(0,np.pi)
        plt.show()

def ej_3():
    x = [np.loadtxt(archivos[0], dtype=float) ,np.loadtxt(archivos[1], dtype=float)]
    P=13
    Titles=['PSD - Señal Sintética vs Real - EEG ojos abiertos','PSD - Señal Sintética vs Real - EEG ojos cerrados']
    for i in range(len(x)):
        a, G=ar_model(x[i],P)
        

        x_g=sintex_EEG(a,G,len(x[i]))
        S_g=welch_metod_PSD(x_g,50,'hamming',80)
    
        w_1=np.linspace(0,2*np.pi,len(S_g))
        S_t=PSD_model(a,G,w_1)
        plt.plot(w_1,10*np.log10(S_g), label='Welch Sintética')
        plt.plot(w_1,10*np.log10(S_t), label='PSD Teórica P=13')

        plt.title(Titles[i])
        plt.legend()
        plt.grid()
        plt.xlabel('w [rad/s]')
        plt.ylabel('PSD - Periodorama(Wlech) [dB]')
        plt.xlim(0,np.pi)
        plt.show()

def ej_4():
    x = [np.loadtxt(archivos[0], dtype=float) ,np.loadtxt(archivos[1], dtype=float)]
    P=13
    fs=200

    delta_s=0.001
    delta_p=0.01
    M=[261,269,269,263,259]
    A=[[1,0],[0,1,0],[0,1,0],[0,1,0],[0,1]]
    F=[[0,3,5,200*0.5],[0,3,5,8,10,200*0.5],[0,8,10,13,15,200*0.5],[0,13,15,29,31,200*0.5],[0,29,31,200*0.5]]
    V=[[1/delta_p,1/delta_s],[1/delta_s,1/delta_p,1/delta_s],[1/delta_s,1/delta_p,1/delta_s],[1/delta_s,1/delta_p,1/delta_s],[1/delta_s,1/delta_p]]

    #a)
    H_labels=['Filtro Equirriple - Banda: D','Filtro Equirriple - Banda: T','Filtro Equirriple - Banda: A', 'Filtro Equirriple - Banda: B', 'Filtro Equirriple - Banda: G']
    Margenes=[0.99,1.01,0.001]
    Limites=[[0,10],[0,15],[5,20],[10,35],[25,40]]
    Colores=['b','g','r','c','y']
    H_filts=np.zeros((len(M),len(x[0])))
    w_h=np.zeros((len(M),len(x[0])))
    f=np.linspace(0,100,len(x[0]))

    for i in range(len(M)):
        w_h[i], H_filts[i]=FIR_filter(M[i],A[i],F[i],V[i],fs,len(x[0]))
        plt.plot(f,H_filts[i],label=H_labels[i],color=Colores[i])
        plt.vlines(F[i],ymin=0,ymax=1.1,colors='m',linestyles='--')
        plt.hlines(Margenes,xmin=0,xmax=100,colors='k',linestyles='--')
        plt.title(H_labels[i])
        plt.xlim(Limites[i])
        plt.ylabel('|H| [veces]')
        plt.xlabel('f [Hz]')
        #plt.legend()
        plt.show()
    


    #b)
    X_sintex=np.zeros((len(x),len(x[0])))
    X_real=np.zeros((len(x),len(x[0])))
    for i in range(len(x)):
        a, G=ar_model(x[i],P)
        X_g=fft(sintex_EEG(a,G,len(x[i])),2*len(x[i]))
        X_sintex[i]=X_g[0:len(x[i])]

        X=fft(x[i],2*len(x[i]))
        X_real[i]=X[0:len(x[i])]
    
    pot_real=np.zeros(len(H_filts))
    pot_sintetica=np.zeros(len(H_filts))


    Titles=['Potencia - EEG ojos abiertos','Potencia - EEG ojos cerrados']
    Bandas=['D','T','A','B','G']

    for i in range(len(x)):
        for j in range(len(H_filts)):
            pot_real[j]=potencia(np.abs(X_real[i])*H_filts[j]/len(X_real[i]))
            pot_sintetica[j]=potencia(np.abs(X_sintex[i])*H_filts[j]/len(X_sintex[i]))
        
        n = np.linspace(0,len(H_filts),len(H_filts))
        plt.scatter(n,10*np.log10(pot_real),marker='o', label='Señal Real')
        plt.scatter(n,10*np.log10(pot_sintetica),marker='x',label='Señal Sintética')
        plt.title(Titles[i])
        plt.xticks(n,Bandas)
        plt.legend()
        plt.grid()
        plt.xlabel('Bandas cerebrales')
        plt.ylabel('Potencia - $\\sigma^2$  [dB]')
        plt.show()


        
ej_1()
ej_2()
ej_3() 
ej_4()