import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

from MyCost2 import MyCost

if __name__ == "__main__":
   dsName = "test"
   ds = pd.read_csv('resources/'+dsName+".csv", header=0)
   ds_values = ds.iloc[:,1].values

   if(dsName[0]=='b'):
      # Inverti l'ordine dell'array
      data_inverted = ds_values[::-1]
      # Rimuovi il punto come separatore delle migliaia e sostituisci la virgola con il punto
      data = np.array([float(x.replace(".", "").replace(",", ".")) for x in data_inverted])
   else:
      data = ds_values

   # Algoritmo Pelt con custom cost
   algo = rp.Pelt(custom_cost=MyCost()).fit(data)
   bkps_pelt = algo.predict(pen=100000) # valore coerente con quelli del dataset
   print(f"Num breakpoint {bkps_pelt.__len__()}") # i breakpoint trovati

   # Visualizzazione dei risultati per Pelt
   rp.display(data, bkps_pelt)
   plt.title("BTC Pelt")
   plt.show()

   print("fine")