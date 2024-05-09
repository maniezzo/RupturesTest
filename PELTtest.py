import numpy as np
import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

from MyCost import MyCost

if __name__ == "__main__":
   ds = pd.read_csv('resources/BTC_EURKrakenDatiStorici.csv', header=0)
   ds_values = ds.Ultimo.values

   # Inverti l'ordine dell'array
   data_inverted = ds_values[::-1]

   # Rimuovi il punto come separatore delle migliaia e sostituisci la virgola con il punto
   data = np.array([float(x.replace(".", "").replace(",", ".")) for x in data_inverted])

   # Algoritmo Pelt con custom cost
   algo = rp.Pelt(custom_cost=MyCost()).fit(data)
   bkps_pelt = algo.predict(pen=100000) # valore coerente con quelli del dataset
   print(f"Num breakpoint {bkps_pelt.__len__()}") # i breakpoint trovati

   # Visualizzazione dei risultati per Pelt
   rp.display(data, bkps_pelt)
   plt.title("BTC Pelt")
   plt.show()

   print("fine")