import numpy as np
import matplotlib.pyplot as plt


def Gauss_multivariable_2d(x,y,media,cov):
    k=1/(np.sqrt(np.power(2*np.pi,len(media))*np.linalg.det(cov)))
    

    return fz

def main():
    N=1000

    media=[0,0]
    cov=[[1,0],[0,1]]

    xmin=-4
    xmax=4
    ymin=-4
    ymax=4

    #X=np.random.normal(0,1,N)
    #print(X)
    #Y=np.random.normal(0,1,N)



    x = np.linspace(xmin, xmax, N) # generar N puntos entre xmin y xmax
    y = np.linspace(ymin, ymax, N) # generar N puntos entre ymin e ymax



    XX, YY = np.meshgrid(x, y) # matrices de puntos para x e y
    fz=Gauss_multivariable_2d(XX,YY,media,cov)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, fz, cmap='viridis') # gráfico de superficie fz
    plt.show()
    plt.contour(XX, YY, fz, levels=10, cmap='viridis') # gráfico de n curva
    plt.show()



main()
