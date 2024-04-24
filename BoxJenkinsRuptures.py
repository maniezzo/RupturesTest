import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv('./resources/BoxJenkins.csv', header=0)
data = ds.Passengers.values

# Algoritmo Pelt
pelt = rp.Pelt(model="l2").fit(data)
bkps_pelt = pelt.predict(pen=1000)

# Algoritmo BinarySeg
algo = rp.Binseg(model="l2").fit(data)
bkps_binSeg = algo.predict(pen=1000)

# Algoritmo Dynp
algo = rp.Dynp(model="l2").fit(data)
bkps_dynp = algo.predict(n_bkps=18)

# Visualizzazione dei risultati per Pelt
rp.display(data, bkps_pelt)
plt.title("BoxJenkins Pelt")
plt.show()

# Visualizzazione dei risultati per BinarySeg
rp.display(data, bkps_binSeg)
plt.title("BoxJenkins BinarySeg")
plt.show()

# Visualizzazione dei risultati per Dynp
rp.display(data, bkps_dynp)
plt.title("BoxJenkins Dynp")
plt.show()
