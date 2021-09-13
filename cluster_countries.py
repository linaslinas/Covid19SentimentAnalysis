import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn
from plotly import graph_objs as go
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, plot

covid_data = pd.read_csv("COVID.csv")

# Get total number of confirmed cases for each country
confirmed_cases = covid_data.groupby("Country/Region")["Confirmed"].sum()

# Get total number of death cases for each country
death_cases = covid_data.groupby("Country/Region")["Deaths"].sum()

cleaned_data = pd.concat([confirmed_cases, death_cases], axis=1)

cleaned_data['fatality_ratio'] = cleaned_data['Deaths']/cleaned_data['Confirmed']
cleaned_data['fatality_ratio'] = cleaned_data['fatality_ratio']*100

cleaned_data = cleaned_data.reset_index()

assess_countries = np.array(cleaned_data[['fatality_ratio' ,'Confirmed']])

kmeans = KMeans(n_clusters=3, max_iter=100).fit(assess_countries)
risk_group = kmeans.labels_
centroids = kmeans.cluster_centers_
SSE = kmeans.inertia_
print("sum of square error: ", SSE)

cleaned_data['risk_group'] = risk_group

# Write clustered data to csv
cleaned_data.to_csv('risk_groups.csv')

# Plot the results from clustering 
risk_group_0 = cleaned_data[cleaned_data["risk_group"] == 0]
risk_group_1 = cleaned_data[cleaned_data["risk_group"] == 1]
risk_group_2 = cleaned_data[cleaned_data["risk_group"] == 2]

print(risk_group_0)
print(risk_group_1)
print(risk_group_2)

plt.plot(risk_group_0['fatality_ratio'], risk_group_0['Confirmed'], 'o', label="Low Risk Countries", color='green')
plt.plot(risk_group_1['fatality_ratio'], risk_group_1['Confirmed'],  'o', label="High Risk Countries", color='orange')
plt.plot(risk_group_2['fatality_ratio'], risk_group_2['Confirmed'],  'o', label="Moderate Risk Countries", color='blue')

plt.setp(plt.plot(centroids[0][0], centroids[0][1], 'kx', label="Centroids for each Risk Group", color='black'), mew=2.0)
plt.setp(plt.plot(centroids[1][0], centroids[1][1], 'kx', color='black'), mew=2.0)
plt.setp(plt.plot(centroids[2][0], centroids[2][1], 'kx', color='black'), mew=2.0)

plt.legend(loc="upper right")
plt.xlabel('Case Fatality Rate (%)')
plt.ylabel('Number of Confirmed Cases')
plt.title('Cluster on Case Fatality Rate and Confirmed Cases')
plt.show()


# Build a heatmap showing the different COVID risk levels of each country.
risk_labels = cleaned_data['risk_group'].to_numpy()

alter_risk_labels = []
for i in range(len(risk_labels)):
	alter_risk_labels.append([risk_labels[i]])

countries = cleaned_data['Country/Region'].to_numpy()


seaborn.heatmap(alter_risk_labels, square=True)
data_risk_label = dict(type = 'choropleth', 
           	locations = countries,
           	locationmode = 'country names',
           	z = risk_labels, 
           	text = countries,
           	colorbar = {'title':'2 = Moderate Risk, 0 = Low Risk, 1 = High Risk'})
layout_risk_label = dict(title = 'COVID-19 Risk Level Clusters (Feb 1,2020 - June 30,2020)', geo = dict(showframe = False, projection = {'type': 'mercator'}))
choromap1 = go.Figure(data = [data_risk_label], layout=layout_risk_label)
plot(choromap1)



# Build a heatmap showing the number of confirmed cases in different countries around the world.
confirmed_cases = cleaned_data['Confirmed'].to_numpy()

covid_cases = []
for i in range(len(confirmed_cases)):
	covid_cases.append([confirmed_cases[i]])


seaborn.heatmap(covid_cases, square=True)
data_cases = dict(type = 'choropleth', 
           	locations = countries,
           	locationmode = 'country names',
           	z = confirmed_cases, 
           	text = countries,
           	colorbar = {'title':'Number of Confirmed Cases'})
layout_cases = dict(title = 'COVID-19 Confirmed Cases Per Country (Feb 1,2020 - June 30,2020)', geo = dict(showframe = False, projection = {'type': 'mercator'}))
choromap2 = go.Figure(data = [data_cases], layout=layout_cases)
plot(choromap2)