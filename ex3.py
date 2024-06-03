import numpy as np
import matplotlib.pyplot as plt

# Função para calcular a média amostral de N variáveis aleatórias normais
def sample_mean(N, num_experiments):
    means = np.zeros(num_experiments)
    for i in range(num_experiments):
        samples = np.random.normal(0, 1, N)
        means[i] = np.mean(samples)
    return means

# Número de experimentos
num_experiments = 10000

# Valores de N para testar
N_values = [10, 50, 100, 500]

# Definindo limites comuns para o eixo x
x_limits = (-1.3, 1.3)

# Plotando os histogramas
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

for i, N in enumerate(N_values):
    row = i // 2
    col = i % 2
    means = sample_mean(N, num_experiments)
    axs[row, col].hist(means, bins=30, density=True, alpha=0.6, color='g')
    axs[row, col].set_title(f'Histograma da média amostral para N={N}')
    axs[row, col].set_xlabel('Média amostral')
    axs[row, col].set_ylabel('Frequência')
    axs[row, col].axvline(0, color='r', linestyle='dashed', linewidth=1)
    axs[row, col].set_xlim(x_limits)  # Ajustando os limites do eixo x

    
plt.subplots_adjust(hspace=0.25)
plt.tight_layout()
plt.show()