
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

NYC["RESORGANICSTONS"].hist(bins = 20)
plt.title("Recylce")
plt.xlabel("Organic recycle Collection")
plt.ylabel("# of collection")

NYC["PAPERTONSCOLLECTED"].hist(bins = 20)
plt.title("Recylce")
plt.xlabel("Paper and Cardboard recycle Collection")
plt.ylabel("# of collection")

#Project Milestone 2

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import matplotlib
%matplotlib inline

x = NYC2[["PAPERTONSCOLLECTED", "MGPTONSCOLLECTED","RESORGANICSTONS", "SCHOOLORGANICTONS", "LEAVESORGANICTONS","XMASTREETONS"]]
y = NYC2["REFUSETONSCOLLECTED"]

#Decision Tree

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

reg2 = DecisionTreeRegressor(max_depth = 2)
reg2 = reg2.fit(x_train, y_train)

pred_2 = reg2.predict(x_test)

mean_squared_error(pred_2, y_test)

reg2 = DecisionTreeRegressor(max_depth = 2)
reg2 = reg2.fit(x_train, y_train)

pred_2 = reg2.predict(x_test)

mean_squared_error(pred_2, y_test)

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

# Project Milestone 3 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import folium

#Two new graph

sns.jointplot(x = "REFUSETONSCOLLECTED", y = "PAPERTONSCOLLECTED", kind = "kde", color = "Green",data = NYC2)

sns.jointplot(x = "REFUSETONSCOLLECTED", y = "MGPTONSCOLLECTED", kind = "hex", color = "red",data = NYC2)

map = folium.Map(location=[40.8747, -73.8951])

folium.Marker([40.75, -73.86667], popup="Queens", tooltip = "Click for information",
             icon = folium.Icon(color = 'orange')).add_to(map)
folium.Marker([40.826153, -73.920265], popup="Bronx", tooltip = "Click for information",
             icon = folium.Icon(color = "red")).add_to(map)
folium.Marker([40.75325, -74.00381], popup="Manhattan", tooltip = "Click for information",
             icon = folium.Icon(color = "green")).add_to(map)
folium.Marker([40.692528, -73.991], popup="Brooklyn", tooltip = "Click for information",
             icon = folium.Icon(color = "blue")).add_to(map)
folium.Marker([40.580753, -74.152794], popup="Staten Island", tooltip = "Click for information",
             icon = folium.Icon(color = "purple")).add_to(map)
map

# Clusters

kmeans = KMeans(n_clusters = 5)
kmeans.fit(x)
clusters = kmeans.predict(x)

NYC2["clusters"] = clusters
NYC2.head()

sns.pairplot(NYC2, hue = "clusters")

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)
x_scaled

kmeans_scaled = KMeans(n_clusters = 5)
kmeans_scaled.fit(x_scaled)

clusters_scaled = kmeans_scaled.predict(x_scaled)
clusters_scaled

NYC2["clusters_scaled"] = clusters_scaled
NYC2.head()

sns.pairplot(NYC2, hue = "clusters_scaled")

cluster_scaled_map = {"0":"Queens", "1":"Brooklyn", "2":"Staten Island", "3":"Bronx", "4":"Manhattan"}
NYC2["mapped_clusters_scaled"] = NYC2["clusters_scaled"].apply(str).map(cluster_scaled_map)
NYC2.head()

# Correlation, causation, and heat maps

NYC2.corr()

corr_matrix = NYC2.corr()
sns.heatmap(corr_matrix)

NYC.corr()

corr_matrix2 = NYC.corr()
sns.heatmap(corr_matrix2)
