
# coding: utf-8

#Project milestone 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


NYC = pd.read_csv("DSNY_Monthly_Tonnage_Data.csv")

NYC['PAPERTONSCOLLECTED'] = NYC['PAPERTONSCOLLECTED'].replace(np.nan, 0)
NYC['MGPTONSCOLLECTED'] = NYC['MGPTONSCOLLECTED'].replace(np.nan, 0)
NYC['RESORGANICSTONS'] = NYC['RESORGANICSTONS'].replace(np.nan, 0)
NYC['SCHOOLORGANICTONS'] = NYC['SCHOOLORGANICTONS'].replace(np.nan, 0)
NYC['LEAVESORGANICTONS'] = NYC['LEAVESORGANICTONS'].replace(np.nan, 0)
NYC['XMASTREETONS'] = NYC['XMASTREETONS'].replace(np.nan, 0)

NYC

Month_2020 = NYC["MONTH"] > "2020"
Month_2020_3 = NYC["MONTH"] < "2020 / 03"

NYC2 = NYC[Month_2020 & Month_2020_3]
NYC2.head(10)

NYC2.dtypes

sns.pairplot(data = NYC2, hue = "BOROUGH")

#Updated
sns.relplot(x = "REFUSETONSCOLLECTED", y = "MGPTONSCOLLECTED", hue = "BOROUGH", data = NYC2)
plt.title("Recycle")
plt.xlabel("Garbage Collection")
plt.ylabel("Recyle trash glass, paper, and plastic")

#Updated
sns.relplot(x = "REFUSETONSCOLLECTED", y = "PAPERTONSCOLLECTED", hue = "BOROUGH", data = NYC2)
plt.title("Recycle")
plt.xlabel("Garbage Collection")
plt.ylabel("Recyle trash paper, cardboard, and aluminum") 

#Single Variables
NYC["REFUSETONSCOLLECTED"].hist(bins = 20)
plt.title("Recylce")
plt.xlabel("Garbage Collection")
plt.ylabel("# of collection")

NYC["MGPTONSCOLLECTED"].hist(bins = 20)
plt.title("Recylce")
plt.xlabel("Metal, Glass, Plastic recycle Collection")
plt.ylabel("# of collection")

#Project Milestone 2

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
import matplotlib
%matplotlib inline

x = NYC2[["PAPERTONSCOLLECTED", "MGPTONSCOLLECTED","RESORGANICSTONS", "SCHOOLORGANICTONS", "LEAVESORGANICTONS","XMASTREETONS"]]
y = NYC2["REFUSETONSCOLLECTED"]

#Decision Tree

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

reg5 = DecisionTreeRegressor(max_depth = 5)
reg5 = reg5.fit(x_train, y_train)

pred_5 = reg5.predict(x_test)

mean_squared_error(pred_5, y_test)

reg10 = DecisionTreeRegressor(max_depth = 10)
reg10 = reg10.fit(x_train, y_train)

pred_10 = reg10.predict(x_test)

mean_squared_error(pred_10, y_test)

#Linear Regression

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)

y_test_preds_linear = linear_model.predict(x_test)

mean_squared_error(y_test_preds_linear,y_test)

x2 = NYC2[["PAPERTONSCOLLECTED", "MGPTONSCOLLECTED","RESORGANICSTONS"]]
y2 = NYC2["REFUSETONSCOLLECTED"]

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size = 0.2)

linear_model2 = LinearRegression()
linear_model2.fit(x_train2,y_train2)

y_test_preds_linear2 = linear_model2.predict(x_test2)

mean_squared_error(y_test_preds_linear2,y_test2)
