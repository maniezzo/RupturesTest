import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv('resources/PTemp_C_Avg.csv', header=0)
data = ds.PTemp_C_Avg.values

# Algoritmo Pelt
pelt = rp.Pelt(model="l2").fit(data)
bkps_pelt = pelt.predict(pen=10)

# Algoritmo BinarySeg
algo = rp.Binseg(model="l2").fit(data)
bkps_binSeg = algo.predict(pen=10)


# Algoritmo Dynp
fig, ax = plt.subplots(2,3, figsize=(1280/96, 720/96), dpi=96)
ax = ax.ravel()

algo = rp.Dynp(model="l2").fit(data)

for i, n_bkps in enumerate([4, 5, 6, 7, 8, 9]):
    result = algo.predict(n_bkps=n_bkps)
    ax[i].plot(data)
    for bkp in result:
        ax[i].axvline(x=bkp, color='k', linestyle='--')
    ax[i].set_title(f"Dynp model with {n_bkps} breakpoints")

# Visualizzazione dei risultati per Pelt
rp.display(data, [], bkps_pelt)
plt.title("PTemp Pelt")
plt.show()

# Visualizzazione dei risultati per BinarySeg
rp.display(data, [], bkps_binSeg)
plt.title("PTemp BinarySeg")
plt.show()
