import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, bernoulli

def inferencia_bayesiana(amostra, ab_list):

    mu = np.linspace(0, 1, 100)
    n_linhas = len(ab_list)
    fig, ax = plt.subplots(n_linhas, 3, figsize=(18, 10))

    for j in range(n_linhas):
        a0, b0 = ab_list[j]
        # inicialização
        priori = beta.pdf(mu, a0, b0)
        ax[j,0].plot(mu,priori)
        ax[j,0].set_title(f"Priori para $a_0$={a0} e $b_0$={b0}")
        
        # Atualizações
        for i in range(len(amostra)):
            sub_amostra = amostra[:i+1]
            n_cara = sum(sub_amostra)
            n_coroa = (i+1) - n_cara
            a = a0 + n_cara
            b = b0 + n_coroa
            if i==0:
                likelihood = (mu**n_cara) * ((1 - mu)**n_coroa)
            else:
                likelihood *= (mu**n_cara) * ((1 - mu)**n_coroa)
            posteriori = beta.pdf(mu, a, b)

            # plot
            ax[j,1].plot(mu,likelihood, label=f"após {i+1} jogada(s)")
            ax[j,1].set_title(f"Likelihood")
            ax[j,1].legend()
            ax[j,2].plot(mu,posteriori, label=f"após {i+1} jogada(s)")
            ax[j,2].set_title(f"Posteriori")
            ax[j,2].legend()


    fig.suptitle(f'Amostra={amostra}', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # amostra
    mu_true = 0.7
    n = 5
    amostra = bernoulli.rvs(mu_true, size=n)
    ab_list = [(1,1),(2,2)]

    inferencia_bayesiana(amostra, ab_list)


    

    
