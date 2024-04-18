import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# example of a difference transform (python)
# difference dataset
def difference(data, interval):
   return [data[i] - data[i - interval] for i in range(interval,len(data))]

# invert difference
def invert_difference(orig_data, diff_data, interval):
   inverted = [diff_data[i]+orig_data[i] for i in range(interval)]
   for i in range(interval,len(diff_data)):
      inverted = np.append(inverted,diff_data[i]+inverted[i-interval])
   return inverted

def main():
   # define dataset
   df = pd.read_csv("FilRouge.csv")
   data = df.iloc[:,1].values
   print("data: ",data)

   # difference 1 period
   diff1 = difference(data, 1)
   print("diff1: ",diff1)

   # difference 4 period
   diff4 = difference(diff1, 4)
   print("diff4: ",diff4)

   # forecast 4 periods
   diff4 = np.append(diff4,[0]*4)

   # invert difference 4
   inv4 = invert_difference(diff1[:4], diff4, 4)
   inv4 = np.append(diff1[:4],inv4)
   print(" inv4: ",inv4)

   # invert difference 1
   inv1 = invert_difference(data[:1], inv4, 1)
   inv1 = np.append(data[:1],inv1)
   print(" inv1: ",inv1)

   plt.figure(figsize=(9,6))
   plt.plot(inv1,linewidth=5,label="predict/forecast")
   plt.plot(data,label="data")
   plt.plot(range(5,len(diff4)+5),diff4,label="diff4")
   plt.legend()
   plt.show()
   return

if __name__ == "__main__":
   main()
   print("... exiting")