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

   # Algoritmo BinarySeg
   #algo = rp.Binseg(model="l1").fit(data)
   algo = rp.Binseg(custom_cost=MyCost()).fit(data)
   bkps_binSeg = algo.predict(pen=100000)
   print(f"Num breakpoint {bkps_binSeg.__len__()}")

   # Visualizzazione dei risultati per BinarySeg
   rp.display(data, bkps_binSeg)
   plt.title("BTC BinarySeg")
   plt.show()