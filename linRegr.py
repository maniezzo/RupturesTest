import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create a DataFrame with hours studied and exam scores
df = pd.DataFrame({
    'hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'score': [64, 66, 76, 73, 74, 81, 83, 82, 80, 88, 84, 82, 91, 93, 89]
})

# Define predictor and response variables
y = df['score']
x = df['hours']

# Add a constant term to the predictor variable
x = sm.add_constant(x)

# Fit the linear regression model
model = sm.OLS(y, x).fit()

# View the model summary
print(model.summary())

# Scatter plot of data points
plt.scatter(df['hours'], df['score'], color='blue', label='Data points')

# Plot the regression line
plt.plot(df['hours'], model.predict(x), color='red', label='Regression line')

plt.title('OLS Regression: Exam Scores vs. Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()
