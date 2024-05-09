import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

from MyCost import MyCost
from MyCost2 import MyCost

if __name__ == "__main__":
   ds = pd.read_csv('resources/BTC_EURKrakenDatiStorici.csv', header=0)
   ds_values = ds.Ultimo.values

   # Inverti l'ordine dell'array
   data_inverted = ds_values[::-1]

   # Rimuovi il punto come separatore delle migliaia e sostituisci la virgola con il punto
   data = np.array([float(x.replace(".", "").replace(",", ".")) for x in data_inverted])

   # Algoritmo Dynp
   fig, ax = plt.subplots(2, 3, figsize=(1280 / 96, 720 / 96), dpi=96)
   ax = ax.ravel()

   algo = rp.Dynp(custom_cost=MyCost()).fit(data)

   for i, n_bkps in enumerate([6, 7, 8, 9, 10, 24]):
      result = algo.predict(n_bkps=n_bkps)
      ax[i].plot(data)
      for bkp in result:
         ax[i].axvline(x=bkp, color='k', linestyle='--')
      ax[i].set_title(f"Dynp model with {n_bkps} breakpoints")
      print(f"Dynp model with {n_bkps} breakpoints")


   # Visualizzazione dei risultati per DYNP
   rp.display(data, result)
   plt.title("BTC DYNP")
   plt.show()