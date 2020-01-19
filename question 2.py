# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:28:07 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('annual_temp.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (annual temp)')
plt.xlabel('YEAR')
plt.ylabel('MEAN')
plt.show()