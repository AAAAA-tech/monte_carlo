from scipy.stats import levy_stable
import numpy as np
from matplotlib import pyplot as plt

# Latex font in figures
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def monte_carlo_test(mon_car_tr, overlap_per):

    # input params
    alpha, beta = 1.7, .0
    # generate prices
    price = levy_stable.rvs(
        alpha,
        beta,
        size=mon_car_tr
    )

    # Generate 1 day returns
    one_day_returns = [
        (price[i+1]-price[i])/price[i]
        for i in range(len(price)-1)
    ]

    # Generate overlapping data
    overlap_data = []
    for i in range(mon_car_tr-overlap_per):
        val = 0
        for j in range(overlap_per):
            val = val + one_day_returns[i+j]
        overlap_data.append(val)

    # Probability of specific value in oveerlappig data
    probs = [
        levy_stable.ppf(i, alpha, beta)
        for i in overlap_data
    ]

    return probs, overlap_data

if __name__ == "__main__":

    mon_car_tr = 750   # Monte - Carlo trials
    overlap_per = 10   # Overlapping period

    probs, overlap_data = monte_carlo_test(mon_car_tr, overlap_per)

    fig,ax=plt.subplots()
    ax.scatter(probs, overlap_data)
    ax.set_xlabel('The values')
    ax.set_ylabel('Probability')
    plt.show()
