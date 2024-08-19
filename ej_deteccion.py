import numpy as np
import matplotlib.pyplot as plt

def generate_ar1_noise(a, sigma2, M):
    # Generar ruido blanco W(n) con varianza sigma2
    W = np.random.normal(0, np.sqrt(sigma2), M)
    # Inicializar el ruido AR1
    V = np.zeros(M)
    V[0] = W[0]
    for n in range(1, M):
        V[n] = a * V[n-1] + W[n]
    return V

def apply_matched_filter(signal, noise, N):
    # Diseñar el filtro adaptado (respuesta al impulso invertida)
    h = signal[:N][::-1]
    # Aplicar el filtro usando la convolución
    filtered_output = np.convolve(noise, h, mode='same')
    return filtered_output

# Parámetros
a_values = [0.5, 3]
sigma2_values = [0.1, 1.0]
N = 10
M = 100

# Señal determinística x(n) = 1, 1 ≤ n ≤ N
signal = np.ones(N)

for a in a_values:
    for sigma2 in sigma2_values:
        # Generar ruido AR1
        noise = generate_ar1_noise(a, sigma2, M)
        # Superponer señal al ruido
        superposed_signal = noise.copy()
        superposed_signal[:N] += signal
        # Aplicar filtro adaptado
        filtered_output = apply_matched_filter(signal, superposed_signal, N)
        
        # Graficar
        plt.figure(figsize=(12, 6))
        
        # Gráfico de la señal, ruido y señal superpuesta al ruido
        plt.subplot(2, 1, 1)
        plt.plot(noise, label='Ruido AR1')
        plt.plot(superposed_signal, label='Señal superpuesta al ruido')
        plt.axvline(x=N-1, color='r', linestyle='--', label='Fin de la señal')
        plt.title(f'a = {a}, sigma^2 = {sigma2}')
        plt.legend()
        
        # Gráfico de la salida del filtro
        plt.subplot(2, 1, 2)
        plt.plot(filtered_output, label='Salida del filtro adaptado')
        plt.title('Salida del filtro adaptado')
        plt.legend()
        
        plt.tight_layout()
        plt.show()