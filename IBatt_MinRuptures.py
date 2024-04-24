import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv('resources/IBatt_Min.csv', header=0)
data = ds.IBatt_Min.values

# Converti le stringhe in numeri interi
#print(data)

# Algoritmo Pelt
pelt = rp.Pelt(model="l2").fit(data)
bkps_pelt = pelt.predict(pen=0.00005)

# Algoritmo BinarySeg
algo = rp.Binseg(model="l2").fit(data)
bkps_binSeg = algo.predict(pen=0.00005)

# Algoritmo Dynp
algo = rp.Dynp(model="l2").fit(data)
bkps_dynp = algo.predict(n_bkps=15)
# Visualizzazione dei risultati per Pelt
rp.display(data, bkps_pelt)
plt.title("ETH Pelt")
plt.show()

# Visualizzazione dei risultati per BinarySeg
rp.display(data, bkps_binSeg)
plt.title("ETH BinarySeg")
plt.show()

# Visualizzazione dei risultati per Dynp
rp.display(data, bkps_dynp)
plt.title("ETH Dynp")
plt.show()
