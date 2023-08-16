
# Step -1 Import data files
import numpy as np
import pandas as pd

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Simple Linear Regression\\delivery_time.csv")
df
df.head()

# Step - 2 Split the variables in X and Y
Y = df["Delivery Time"]
X = df[["Sorting Time"]]

#EDA
 # Scatter Plot
import matplotlib.pyplot as plt
plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

# Boxplot
plt.boxplot(df["Delivery Time"])
plt.boxplot(df["Sorting Time"])

# Histogram
plt.hist(df["Delivery Time"],bins=5)
plt.hist(df["Sorting Time"],bins=5)


# Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_ #Bo
LR.coef_ #B1

# Predict the value
Y_pred = LR.predict(X)
Y_pred

# Scatter Plot with Plot
plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],Y_pred,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

# Then Finding Error 
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y, Y_pred)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(Y, Y_pred).round(3)*100)





"""
There is a Difference of RMSE is 2.792 and the R2 is 0.682
"""












    

