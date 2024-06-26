import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

from MyCost import MyCost

ds = pd.read_csv('resources/ThermTemp1_Avg.csv', header=0)
data = ds.ThermTemp1_Avg.values

# Algoritmo Pelt con custom cost
algo = rp.Pelt(custom_cost=MyCost()).fit(data)
bkps_pelt = algo.predict(pen=3)

# Algoritmo BinarySeg
algo = rp.Binseg(model="l2").fit(data)
bkps_binSeg = algo.predict(pen=3)

# Algoritmo Dynp
fig, ax = plt.subplots(2,3, figsize=(1280/96, 720/96), dpi=96)
ax = ax.ravel()

algo = rp.Dynp(model="l2").fit(data)

for i, n_bkps in enumerate([6, 7, 8, 9, 10, 11]):
    result = algo.predict(n_bkps=n_bkps)
    ax[i].plot(data)
    for bkp in result:
        ax[i].axvline(x=bkp, color='k', linestyle='--')
    ax[i].set_title(f"Dynp model with {n_bkps} breakpoints")

# Visualizzazione dei risultati per Pelt
rp.display(data, bkps_pelt)
plt.title("ThermTemp Pelt")
plt.show()

# Visualizzazione dei risultati per BinarySeg
rp.display(data, bkps_binSeg)
plt.title("ThermTemp BinarySeg")
plt.show()
