'''
TFbinding Data Operations and Visualization
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

# Read the data by selecting appropriate columns based on column names in the data file
myData = pd.read_csv("TFbinding-HW3-F23.txt", delimiter="\t")

# Calculate the mean and standard deviation of the 'Expression Level' column
expMean = np.mean(myData["Expression Level"])
expStd = np.std(myData["Expression Level"])
print(f"Mean of Expression Level: {expMean}")
print(f"Standard Deviation of Expression Level: {expStd}")

# Calculate the total binding score, and calculate its mean and standard deviation
myData["SumOfBinding"] = myData.iloc[:, 1:7].sum(axis=1)
sumMean = np.mean(myData["SumOfBinding"])
sumStd = np.std(myData["SumOfBinding"])
print(f"Mean of Total Binding Scores: {sumMean}")
print(f"Standard Deviation of Total Binding Scores: {sumStd}")

# Plot two histograms as subplots: one for expression levels and one for total binding scores
plt.figure(figsize=[12, 5]) # Arranged the size of the plots

# Expression Levels histogram
plt.subplot(121)
plt.hist(myData["Expression Level"], bins=20, color="tomato", edgecolor="black")
plt.axvline(expMean, color='k', linestyle='dashed', linewidth=2, label=f'Mean: {round(expMean, 3)}')
plt.axvline(expMean + expStd, color='b', linestyle='dashed', linewidth=2, label=f'Std Dev: {round(expStd, 3)}')
plt.axvline(expMean - expStd, color='b', linestyle='dashed', linewidth=2)
plt.xlabel("Expression Levels")
plt.ylabel("Frequencies")
plt.title("Histogram of Expressions")
plt.legend()

# Total Binding Scores histogram
plt.subplot(122)
plt.hist(myData["SumOfBinding"], bins=20, color="tomato", edgecolor="black")
plt.axvline(sumMean, color='k', linestyle='dashed', linewidth=2, label=f'Mean: {round(sumMean, 3)}')
plt.axvline(sumMean + sumStd, color='b', linestyle='dashed', linewidth=2, label=f'Std Dev: {round(sumStd, 3)}')
plt.axvline(sumMean - sumStd, color='b', linestyle='dashed', linewidth=2)
plt.xlabel("Total Binding Scores")
plt.ylabel("Frequencies")
plt.title("Histogram of Binding Scores")
plt.legend()
plt.show()

# Set input data and initial guesses for linear regression model
xi = myData["SumOfBinding"]
yi_exp = myData["Expression Level"]
expMean, sumMean = np.mean(yi_exp), np.mean(xi)
p0 = [expMean, sumMean]

# Define the error function
def residuals(p, y, x):
    a, b = p
    error = y - (a + b * x)
    return error

# Use leastsq function to find linear regression coefficients
coef, _ = leastsq(residuals, p0, args=(yi_exp, xi))
a, b = coef[0], coef[1]
print("Fitted Equation: Expression Level =", round(a, 3), "+", round(b, 3), "* Total Binding Score")

# Predict gene expression level using the linear regression equation obtained
yi_est = a + b * xi
error = yi_exp - yi_est  # Error = True_expression_level – Predicted_expression_level for a gene (calculate error)
print("Predicted Expression Levels:", yi_est)
print("Prediction Errors:", error)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(xi, yi_exp, marker='o', label='True Expression Level') #True expression values (circles)
plt.plot(xi, yi_est, label='Fitted Equation', color='red') # predictions (line)
plt.scatter(xi, error, marker='*', label='Prediction Error', color='green') # errors (asterix)

plt.xlabel('Total Binding Score')
plt.ylabel('Expression Level')
plt.title('Linear Regression for Gene Expression Prediction')
plt.legend()
plt.grid(True)
plt.show()
