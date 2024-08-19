import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




def graficos_item_2b(vector):
        # Crear la figura y la cuadrícula de subgráficos con menos separación
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        # Gráfico en el centro
        ax_center = fig.add_subplot(gs[0, 0])
        ax_center.scatter(vector[0,:],vector[1,:],alpha=0.5)
        ax_center.set_title('Grafico de dispersión Y0 vs Y1')
        ax_center.set_xlabel('Y0')
        ax_center.set_ylabel('Y1')

        # Gráfico a la derecha
        ax_right = fig.add_subplot(gs[0, 1])
        ax_right.hist(vector[1,:], bins=30, orientation='horizontal')
        ax_right.set_title('Histograma Y1')
        ax_right.yaxis.tick_right()  # Mover los ticks del eje y a la derecha
        y_min, y_max = ax_center.get_ylim()
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_xlabel('Cantidad de Muestras')
        ax_right.set_ylabel('Y1')

        # Gráfico abajo
        ax_bottom = fig.add_subplot(gs[1, 0])
        ax_bottom.hist(vector[0,:], bins=30)
        ax_bottom.set_title('Histograma Y0')
        ax_bottom.xaxis.tick_bottom()  # Mover los ticks del eje x abajo
        x_min, x_max = ax_center.get_xlim()
        ax_bottom.set_xlim(x_min, x_max)
        ax_bottom.invert_yaxis()  # Invertir el eje y para que esté boca abajo
        ax_bottom.set_ylabel('Cantidad de Muestras')
        ax_bottom.set_xlabel('Y0')

        plt.show()

def grafico_item_3(mse,CR,nombres):
    for i in range(len(mse)):
        plt.plot(CR,mse[i,:],'-',label=nombres[i])
    plt.title('MSE vs CR[%]')
    plt.xlabel('CR[%]')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

'''ingresa un vector y el tamaño de los segmentos ==> devuelve matriz de los segmentos con los valores consecutivos del vector original'''
def segmentacion(vector, tamaño_segmentos):
    #si el audio es impar le agrego un 0

    if len(vector%2 != 0):
        vector=np.append(vector,0)

    cant_segmentos=int(len(vector)/tamaño_segmentos) #calculo la cantidad de segmentos
    segmentacion=np.zeros((cant_segmentos+1,tamaño_segmentos))

    for i in range(cant_segmentos):
        segmentacion[i]=vector[i*tamaño_segmentos:(i+1)*tamaño_segmentos]
    #MATRIZ Los experimentos son columnas y las filas las VA
    return np.transpose(segmentacion)

'''Recibe vector de VAs==>devuelve una lista con el audio recontruido'''
def reconstuccion_audio(X):
    return X.reshape(-1,order='F')



'''recibe vector de VA==> devuelve la media estimada'''
def estimador_media(X):
    mu=np.zeros(len(X))
    for i in range(len(X[0])):
        mu+=X[:,i]

    return mu/len(X[0])

'''recibe vector VA==> devuelve matriz de covarianza estimada'''
def estimador_matriz_covarianza(X):
    C_x=np.zeros((len(X),len(X)))
    mu=estimador_media(X)
    for i in range(len(X[0])):
        C_x+=np.outer((X[:,i]-mu),(X[:,i]-mu))

    return C_x/len(X[0])


'''Recibe matriz D de Autovect, vector D de Autoval y el Porcentaje de Compresión'''
'''Asume que D y V están ordenadas del max al min'''
def PCA(V,D,Comp_rate):
    cant_autoval_a_conservar=int((1-(Comp_rate/100))*len(D))

    U=np.zeros((len(V),cant_autoval_a_conservar))

    for i in range(cant_autoval_a_conservar):
        U[:,i]=V[:,i]

    return U

'''Recibe dos señales==> Devuelve su error'''
def MSE(señal,señal_reconstruida):
    mse=0
    for i in range(len(señal)):
        mse+=pow((señal[i]-señal_reconstruida[i]),2)

    return mse/len(señal)


def ej_1():
    #Definimos parametros y extraemos audio normalizado
    L=2
    audio, freq = sf.read('audio_01_2024a.wav')
    norma=np.linalg.norm(audio)
    audio_norm= audio / norma
    #Segmentamos y fabricamos vector X de VAs
    X=segmentacion(audio_norm,L)
    graficos_item_2b(X)
    #Estimamos C_x
    C_x=estimador_matriz_covarianza(X)
    print(C_x)

    D , V = np.linalg.eig(C_x)

    #Se aplica transformación U^t*X (EN ESTE EJ NO SE APLICA COMPRESION)
    Y=np.dot(np.transpose(V),X)
    graficos_item_2b(Y)
    '''acá termina el ej1'''



def ej_2():
    #Definimos parametros y extraemos audio normalizado
    L=1323
    audio, freq=sf.read('audio_02_2024a.wav')
    norma=np.linalg.norm(audio)
    audio_norm= audio / norma
    CR=[70,90,95]

    #Segmentamos y fabricamos vector X de VAs
    X=segmentacion(audio_norm,L)

    #Estimamos C_x y Diagonalizamos
    C_x = estimador_matriz_covarianza(X)

    D , V = np.linalg.eig(C_x)
    print(D)

    #Reproducir audio normal
    sd.play(audio,freq)
    sd.wait()

    #Reproducir el audio para cada Comp_rate, aplicamos PCA
    for i in range(len(CR)):
        U=PCA(V,D,CR[i])
        Y=np.dot(np.transpose(U),X)
        X_reconstuida=np.dot(U,Y)
        audio_r=reconstuccion_audio(X_reconstuida)*norma
        sd.play(audio_r,freq)
        sd.wait()
    '''termina ejercicio 2'''

def ej_3():
    CR=[10,20,30,40,50,60,70,80,90,99]
    audios=['audio_01_2024a.wav','audio_02_2024a.wav','audio_03_2024a.wav','audio_04_2024a.wav','audio_05_2024a.wav','audio_06_2024a.wav']
    L=1323
    mse=np.zeros((len(audios),len(CR)))
    for i in range(len(audios)):
        audio, freq=sf.read(audios[i])

        norma=np.linalg.norm(audio)
        audio_norm= audio / norma

        X=segmentacion(audio_norm,L)

        C_x = estimador_matriz_covarianza(X)
        D , V = np.linalg.eig(C_x)

        for j in range(len(CR)):
            U=PCA(V,D,CR[j])

            Y=np.dot(np.transpose(U),X)

            X_reconstuida=np.dot(U,Y)

            audio_r=reconstuccion_audio(X_reconstuida)

            mse[i][j] = MSE(audio,audio_r)

    print(mse)
    grafico_item_3(mse,CR,audios)

ej_1()
#ej_2()
#ej_3()
