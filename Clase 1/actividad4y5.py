import numpy as np
import matplotlib.pyplot as plt

'''Recibe 2 vectores ==> devuelve la covarianza'''
def covarianza(vector1,vector2):
    covarianza=0
    media1=np.mean(vector1)
    media2=np.mean(vector2)
    for i in range(len(vector1)):
        covarianza+=(vector1[i]-media1)*(vector2[i]-media2)
    return covarianza/len(vector1)


'''X pertenece a 2xM'''
'''recibe vector aleatorio==>devuelve covarianza'''
def matriz_covarianza(ve_a):

    matriz_cov=np.zeros((len(ve_a),len(ve_a)))

    for i in range(len(ve_a)):
        for j in range(len(ve_a)):
            #aprovecho la simetría de la matriz para reducir calculos
            if i<=j:
                cov_ij=covarianza(ve_a[i],ve_a[j])
                matriz_cov[i,j]=cov_ij
                matriz_cov[j,i]=cov_ij

    return matriz_cov


x1 = np.random.rayleigh(3, size=1000)
x2 = np.random.rayleigh(2, size=1000)

B=[[0.6,-0.2],[0.4,0.2]]
#print(B)

H=[[0.6,-0.2],[0.4,0.7]]
#print(H)

ve_x=[x1,x2]
'''
print(matriz_covarianza(ve_x[0],ve_x[1]))
print('Cov(x1,x1)=',np.cov(ve_x[0]))
print('Cov(x2,x2)=',np.cov(ve_x[1]))
print(np.corrcoef(ve_x[0],ve_x[1]))
print(np.cov(ve_x))
'''
#U=BX
ve_u=np.dot(B,ve_x)
print(matriz_covarianza(ve_u))
print(np.dot(np.dot(B,matriz_covarianza(ve_x)),np.transpose(B)))
print(np.cov(ve_u[0],ve_u[1]))
print('Cov(u1,u1)=',np.cov(ve_u[0]))
print('Cov(u2,u2)=',np.cov(ve_u[1]))
print(np.corrcoef(ve_u[0],ve_u[1]))
print(np.cov(ve_u))

#V=HX
ve_v=np.dot(H,ve_x)
'''
print('Cov(v1,v1)=',np.cov(ve_v[0]))
print('Cov(v2,v2)=',np.cov(ve_v[1]))
print(np.corrcoef(ve_v[0],ve_v[1]))
print(np.cov(ve_v))
'''
plt.title('Dispersión X')
plt.plot(ve_x[0],ve_x[1],'o')
plt.show()

plt.title('Dispersión U')
plt.plot(ve_u[0],ve_u[1],'o')
plt.show()

plt.title('Dispersión V')
plt.plot(ve_v[0],ve_v[1],'o')
plt.show()
#print(ve_x)
