import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv('resources/GBP_USDDatiStorici.csv', header=0)
ds_values = ds.Ultimo.values

# Inverti l'ordine dell'array
data_inverted = ds_values[::-1]

# Rimuovi il punto come separatore delle migliaia e sostituisci la virgola con il punto
data = np.array([float(x.replace(".", "").replace(",", ".")) for x in data_inverted])

# Converti le stringhe in numeri interi
#print(data)

# Algoritmo Pelt
pelt = rp.Pelt(model="l2", jump=2)
bkps_pelt = pelt.fit_predict(data, 5)

# Algoritmo BinarySeg
algo = rp.Binseg(model="l2", jump=2).fit(data)
bkps_binSeg = algo.predict(pen=5)

# Algoritmo Dynp
algo = rp.Dynp(model="l2", min_size=4, jump=2).fit(data)
bkps_dynp = algo.predict(n_bkps=5)

# Visualizzazione dei risultati per Pelt
rp.display(data, bkps_pelt)
plt.title("GBP Pelt")
plt.show()

# Visualizzazione dei risultati per BinarySeg
rp.display(data, bkps_binSeg)
plt.title("GBP BinarySeg")
plt.show()

# Visualizzazione dei risultati per Dynp
rp.display(data, bkps_dynp)
plt.title("GBP Dynp")
plt.show()
