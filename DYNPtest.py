import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd
import time

from MyCost import MyCost
from QRMScostClass import QRMScost

def test_dynp(filepath,n_bkps):
   ds = pd.read_csv('e:/My Drive/Ongoing/segmentation/data/home/'+dsName+".csv", header=0)
   ds_values = ds.iloc[:,1].values

   if(dsName[0]=='b'):
      # Inverti l'ordine dell'array, per dataset strani
      data_inverted = ds_values[::-1]
      # Rimuovi il punto come separatore delle migliaia e sostituisci la virgola con il punto
      data = np.array([float(x.replace(".", "").replace(",", ".")) for x in data_inverted])
   else:
      data = ds_values

   tstart = time.process_time()
   algo = rp.Dynp(custom_cost=QRMScost()).fit(data)
   ttot = time.process_time() - tstart
   print(f"CPU time: {ttot:.6f} seconds")

   result = 0
   if(n_bkps < len(data)//5):
      result = algo.predict(n_bkps=n_bkps)
      print(f"Dynp model with {n_bkps} breakpoints")

   return data, result, ttot

if __name__ == "__main__":
   dsName = "BTC-USD"
   n_bkps = 3
   data, result, ttot = test_dynp(dsName,n_bkps)

   q = QRMScost()
   q.fit(data)
   t0=0
   ctot = 0
   for bkp in result:
      c = q.error(t0,bkp)
      ctot += c
      t0=bkp

   fig, ax = plt.subplots(1, 1, figsize=(1280 / 96, 720 / 96), dpi=96)
   ax.plot(data.tolist())
   for bkp in result:
      ax.axvline(x=bkp, color='k', linestyle='--')
   ax.set_title(f"Dynp model with {n_bkps} breakpoints")
   plt.show()

   rp.display(data, result)
   plt.title(dsName+" DYNP")
   plt.show()

   print(f"DYNP, sataset {dsName} costo {ctot} n_brkpoints {n_bkps} t.cpu {ttot}")
