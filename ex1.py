import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, bernoulli

def inferencia_bayesiana(amostra, ab_list):

    x = np.linspace(0, 1, 100)
    n_linhas = len(ab_list)
    fig, ax = plt.subplots(n_linhas, 3, figsize=(18, 6))

    for j in range(n_linhas):
        a, b = ab_list[j]
        # inicialização
        priori = beta.pdf(x, a, b)
        ax[j,0].plot(x,priori)
        ax[j,0].set_title(f"Priori para $a_0$={a} e $b_0$={b}")
        
        # Atualizações
        for i in range(len(amostra)):
            sub_amostra = amostra[:i+1]
            n_cara = sum(sub_amostra)
            n_coroa = (i+1) - n_cara
            a = a + n_cara
            b = b + n_coroa
            if i==0:
                likelihood = (x**n_cara) * ((1 - x)**n_coroa)
            else:
                likelihood *= (x**n_cara) * ((1 - x)**n_coroa)
            posteriori = beta.pdf(x, a, b)

            # plot
            ax[j,1].plot(x,likelihood, label=f"após {i+1} jogada(s)")
            ax[j,1].set_title(f"Likelihood")
            ax[j,1].legend()
            ax[j,2].plot(x,posteriori, label=f"após {i+1} jogada(s)")
            ax[j,2].set_title(f"Posteriori")
            ax[j,2].legend()


    fig.suptitle(f'Amostra={amostra}', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # amostra
    mu = 0.7
    n = 5
    amostra = bernoulli.rvs(mu, size=n)
    ab_list = [(1,1),(2,2)]

    inferencia_bayesiana(amostra, ab_list)


    

    
