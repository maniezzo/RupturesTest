import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

# Carica i dati
ds = pd.read_csv('./resources/BoxJenkins.csv', header=0)
data = ds.Passengers.values

# change point detection
model = "l2"  # "l1", "rbf", "linear", "normal", "ar",...
algo = rp.Binseg(model=model).fit(data)
my_bkps = algo.predict(n_bkps=3)

# show results
rp.show.display(data, my_bkps, figsize=(10, 6))
plt.show()