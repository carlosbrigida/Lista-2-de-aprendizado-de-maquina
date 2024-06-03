import numpy as np
import matplotlib.pyplot as plt

# Função para plotar histogramas
def plot_histogram(means, N, title, ax):
    ax.hist(means, bins=30, edgecolor='k', alpha=0.7)
    ax.set_title(f'{title} (N={N})')
    ax.set_xlabel('Média')
    ax.set_ylabel('Frequência')

# Configurações
num_samples = 10000  # Número de amostras para calcular a média

# Casos de N
Ns = [1, 2, 10]


fig, ax = plt.subplots(2,3)

# Caso 1: Variáveis Aleatórias Uniforme(0,1)
for i in range(len(Ns)):
    uniform_samples = np.random.uniform(0, 1, (num_samples, Ns[i]))
    uniform_means = np.mean(uniform_samples, axis=1)
    plot_histogram(uniform_means, Ns[i], 'Distribuição Uniforme(0,1)', ax[0,i])

# Caso 2: Variáveis Aleatórias Bernoulli(p)
mu = 0.7  # Parâmetro da distribuição Bernoulli
for i in range(len(Ns)):
    bernoulli_samples = np.random.binomial(1, mu, (num_samples, Ns[i]))
    bernoulli_means = np.mean(bernoulli_samples, axis=1)
    plot_histogram(bernoulli_means, Ns[i], f'Distribuição de Bernoulli ($\mu$={mu})', ax[1,i])

plt.tight_layout()
plt.show()
