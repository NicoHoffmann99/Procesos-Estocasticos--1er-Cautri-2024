import numpy as np
import matplotlib.pyplot as plt


def graficar(x,y):
  plt.title('Dispersión')
  plt.plot(x,y,'o')
  plt.show()


def main():
  theeta=np.pi/10
  N=1000
  R=[[np.cos(theeta),-np.sin(theeta)],[np.sin(theeta),np.cos(theeta)]]
  print(R)
  ve_X=[np.random.uniform(0,2,N),np.random.uniform(0,3,N)]
  ve_Y=np.dot(R,ve_X)

  print(np.corrcoef(ve_Y))
  print(np.cov(ve_Y))
  graficar(ve_Y[0],ve_Y[1])

main()
