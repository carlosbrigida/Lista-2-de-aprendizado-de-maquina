import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Parâmetros das distribuições Gaussianas
mu1, sigma1, pi1 = 0.2, 0.1, 0.3
mu2, sigma2, pi2 = 0.7, 0.15, 0.7

# Gerar amostra de dados
N = 50
data = np.hstack([np.random.normal(mu1, sigma1, int(N * pi1)),
                  np.random.normal(mu2, sigma2, int(N * pi2))])

# Valores de delta e largura de banda h
deltas = [0.04, 0.08, 0.25]
bandwidths = [0.005, 0.07, 0.2]

# Plotar 
fig, axs = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=True)
x = np.linspace(-0.2, 1.2, 1000)
model_pdf = pi1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) / (np.sqrt(2 * np.pi) * sigma1) \
           + pi2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2)) / (np.sqrt(2 * np.pi) * sigma2)

for i in range(3):
    axs[i, 0].hist(data, bins=int((max(data) - min(data)) / deltas[i]), density=True, alpha=0.6, color='blue', edgecolor='black')
    axs[i, 0].plot(x, model_pdf, color='green')
    if i == 0:
        axs[i, 0].set_title('Histograma')
    axs[i, 0].text(0.1, 0.9, f'$\Delta$={deltas[i]}', transform=axs[i, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidths[i]).fit(data[:, np.newaxis])
    log_dens = kde.score_samples(x[:, np.newaxis])
    dens = np.exp(log_dens)
    axs[i, 1].plot(x, dens, alpha=0.6, color='black')
    axs[i, 1].fill(x, dens, alpha=0.6, color='blue')
    axs[i, 1].plot(x, model_pdf, color='green')
    if i == 0:
        axs[i, 1].set_title('Kernel Gaussiano')
    axs[i, 1].text(0.1, 0.9, f'h={bandwidths[i]}', transform=axs[i, 1].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Ajustar o layout e mostrar a figura
plt.tight_layout()
plt.show()
