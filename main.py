import ruptures as rp
import matplotlib.pyplot as plt
import pandas as pd

n_samples, n_dims, sigma = 1000, 3, 2
n_bkps = 4  # number of breakpoints
signal, bkps = rp.pw_constant(n_samples, n_dims, n_bkps, noise_std=sigma)

print(bkps)

# detection
algo = rp.Dynp(model="l2").fit(signal)
result = algo.predict(n_bkps=4)

print(result)

# display
rp.display(signal, bkps, result)
plt.show()