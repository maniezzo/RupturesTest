import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from ruptures.base import BaseCost

class QRMScost(BaseCost):
   """Custom cost for quasi RMSE"""
   model = ""
   min_size = 2

   def fit(self, signal):
      """Set the internal parameter."""
      self.signal = signal
      return self

   def error(self, start, end):
      end = end+1  # last instant is included
      y = self.signal[start:end]
      x = range(len(y))

      # Add a constant term to the predictor variable
      x = sm.add_constant(x)
      # Fit the linear regression model
      model = sm.OLS(y, x).fit()

      square_diff = np.power((y - model.predict(x)),2)
      cost = np.sum(square_diff)/np.sqrt(len(y))

      print(f"{start}-{end-1}: {cost}")

      if(cost<0):
         plt.scatter(range(len(y)), y, color='blue', label='Data points')
         plt.plot(range(len(y)), model.predict(x), color='red', label='Regression line')
         plt.title('OLS Regression')
         plt.legend()
         plt.show()

      return cost