import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.lines import Line2D

# Gerar dados para C1 e C2
np.random.seed(0)
C1_data = np.random.normal(loc=-1, scale=1, size=(10, 2))  # 10 pontos para C1
C2_data = np.random.normal(loc=1, scale=1, size=(10, 2))   # 10 pontos para C2

# Gerar novos dados para classificação
unknown_data = np.vstack([
    np.random.normal(loc=-1, scale=1, size=(2, 2)),  # 2 pontos desconhecidos para C1
    np.random.normal(loc=1, scale=1, size=(2, 2))    # 2 pontos desconhecidos para C2
])

# Criar rótulos para os dados conhecidos
labels_C1 = np.full(10, 'C1')
labels_C2 = np.full(10, 'C2')

# Criar rótulos verdadeiros para os dados desconhecidos
unknown_true_labels = np.array(['C1', 'C1', 'C2', 'C2'])

# Combinar dados e rótulos para todos os pontos
X = np.vstack([C1_data, C2_data, unknown_data])
y = np.hstack([labels_C1, labels_C2, unknown_true_labels])

# Configurar o classificador K-NN
k_values = [1, 3, 5, 7]
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[:-4], y[:-4])  # Excluindo os 4 pontos desconhecidos da matriz X e dos rótulos y
    pred = knn.predict(X[-4:])  # Classificar os 4 pontos desconhecidos
    
    # Plotar os dados conhecidos
    axs[i // 2, i % 2].scatter(C1_data[:, 0], C1_data[:, 1], color='red', label='C1')
    axs[i // 2, i % 2].scatter(C2_data[:, 0], C2_data[:, 1], color='blue', label='C2')
    
    # Plotar a classificação dos novos dados
    for j, p in enumerate(pred):
        color = 'red' if p == 'C1' else 'blue'
        marker = 's' if p == y[-4:][j] else 'x'  # Marcar corretamente classificados com 's' e incorretamente com 'x'
        axs[i // 2, i % 2].scatter(X[-4:][j, 0], X[-4:][j, 1], color=color, marker=marker, s=100)  # Aumentar o tamanho para melhor visualização
        axs[i // 2, i % 2].set_title(f'K={k}')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='C1', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='C2', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='x', color='w', label='Classificação Incorreta', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Classificação Correta', markerfacecolor='white', markeredgecolor='black', markersize=10),
]
axs[0,1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
