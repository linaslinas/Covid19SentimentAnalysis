import numpy as np
from scipy.integrate import odeint
from sklearn.cluster import KMeans
import pandas
import matplotlib.pyplot as plt
import time
import warnings
import seaborn as sns
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot

risk_groups = []
country_risk_labels = []
case_fatality_ratio_per_country = []
country_list = []
centroids = []
SSE = []
assess_each_country = []
risk_group_0 = []
risk_group_1 = []
risk_group_2 = []

#read the datasets
df = pandas.read_csv("COVID.csv")

#get the total number of confirmed cases for each country
confirmed_cases_per_country = df.groupby("Country/Region")["Confirmed"].sum()

#get the total number of death cases for each country
death_cases_per_country = df.groupby("Country/Region")["Deaths"].sum()

#calculate the case fatality ratio for each country
for i in range(len(death_cases_per_country)):
	case_fatality_ratio_per_country.append(((death_cases_per_country[i]) / (confirmed_cases_per_country[i])*100))
	assess_each_country.append([case_fatality_ratio_per_country[i], confirmed_cases_per_country[i]])

print(case_fatality_ratio_per_country)
print("--------------------")
print(assess_each_country)


#reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?fbclid=IwAR3ETqmnQXsEjksQpzMnhZuDt--q2axEduA1_7R_ROPOSTfzbBw53ENr61E#sklearn.cluster.KMeans
kmeans = KMeans(n_clusters=3, max_iter=100).fit(assess_each_country)
risk_groups = kmeans.labels_
centroids = kmeans.cluster_centers_
SSE = kmeans.inertia_

print("************************************")
print("sum of square error")
print("************************************")
print(SSE)
print(centroids)
#----------------------------------------get the list of countries and store it---------------------------
#get all the country names
country_list_df = df.loc[0:186, "Country/Region"]

#put all the unique country names into a list called countr_list
for i in range(187):
	country_list.append(country_list_df[i])
#----------------------------------------------------------------------------------------------------------

#-------------------------------match each country with their clustered labels ----------------------------------
for i in range(187):
	country_risk_labels.append([country_list[i], risk_groups[i], assess_each_country[i][0], assess_each_country[i][1]])

for i in range(187):
	if country_risk_labels[i][1] == 0:
		risk_group_0.append(country_risk_labels[i])
	if country_risk_labels[i][1] == 1:
		risk_group_1.append(country_risk_labels[i])
	if country_risk_labels[i][1] == 2:
		risk_group_2.append(country_risk_labels[i])


#---------------------divide the clustered case fatality rate and confirmed cases of each risk group, this will be used for plotting later----------------------
risk_group_0_case_fatality_rate = []
for i in range(len(risk_group_0)):
	risk_group_0_case_fatality_rate.append(risk_group_0[i][2])
risk_group_0_confirm_cases = []
for i in range(len(risk_group_0)):
	risk_group_0_confirm_cases.append(risk_group_0[i][3])

risk_group_1_case_fatality_rate = []
for i in range(len(risk_group_1)):
	risk_group_1_case_fatality_rate.append(risk_group_1[i][2])
risk_group_1_confirm_cases = []
for i in range(len(risk_group_1)):
	risk_group_1_confirm_cases.append(risk_group_1[i][3])

risk_group_2_case_fatality_rate = []
for i in range(len(risk_group_2)):
	risk_group_2_case_fatality_rate.append(risk_group_2[i][2])
risk_group_2_confirm_cases = []
for i in range(len(risk_group_2)):
	risk_group_2_confirm_cases.append(risk_group_2[i][3])
#----------------------------------------------------------------------------------------------------------


#----------------------------write the country and their corresponding risk groups into a csv file---------------------------
# file_0 = open("Risk0NEW.csv", "a")
# for i in risk_group_0:
# 	file_0.write(i[0])
# 	file_0.write("\t")
# 	file_0.write(str(i[1]))
# 	file_0.write("\t")
# 	file_0.write(str(i[2]))
# 	file_0.write("\t")
# 	file_0.write(str(i[3]))
# 	file_0.write("\n")
# file_0.close()

# file_1 = open("Risk1NEW.csv", "a")
# for i in risk_group_1:
# 	file_1.write(i[0])
# 	file_1.write("\t")
# 	file_1.write(str(i[1]))
# 	file_1.write("\t")
# 	file_1.write(str(i[2]))
# 	file_1.write("\t")
# 	file_1.write(str(i[3]))
# 	file_1.write("\n")
# file_1.close()

# file_2 = open("Risk2NEW.csv", "a")
# for i in risk_group_2:
# 	file_2.write(i[0])
# 	file_2.write("\t")
# 	file_2.write(str(i[1]))
# 	file_2.write("\t")
# 	file_2.write(str(i[2]))
# 	file_2.write("\t")
# 	file_2.write(str(i[3]))
# 	file_2.write("\n")
# file_2.close()
#-------------------------------------------------------------------------------------------------------------------------------------
print("***********************************")
print("Low Risk Countries")
print("***********************************")
print(risk_group_0)
print("***********************************")
print("High Risk Countries")
print("***********************************")
print(risk_group_1)
print("***********************************")
print("Moderate Risk Countries")
print("***********************************")
print(risk_group_2)

#---------------------------------------------plotting --------------------------------------------------
#plot the results of clustering
plt.plot(risk_group_0_case_fatality_rate, risk_group_0_confirm_cases, 'o', label="Low Risk Countries", color='green')
plt.plot(risk_group_1_case_fatality_rate, risk_group_1_confirm_cases,  'o', label="High Risk Countries", color='orange')
plt.plot(risk_group_2_case_fatality_rate, risk_group_2_confirm_cases,  'o', label="Moderate Risk Countries", color='blue')

plt.setp(plt.plot(centroids[0][0], centroids[0][1], 'kx', label="Centroids for each Risk Group", color='black'), mew=2.0)
plt.setp(plt.plot(centroids[1][0], centroids[1][1], 'kx', color='black'), mew=2.0)
plt.setp(plt.plot(centroids[2][0], centroids[2][1], 'kx', color='black'), mew=2.0)

plt.legend(loc="upper right")
plt.xlabel('Case Fatality Rate (%)')
plt.ylabel('Number of Confirmed Cases')
plt.title('Cluster on Case Fatality Rate and Confirmed Cases')
plt.show()
#-------------------------------------------------------------------------------------


# -----------------heatmap showing the case fatality rate of each country (Note: uncomment this section to show heatmap, only 1 heatmap can be generated at a time)-----------------------

# #Reference: https://www.kaggle.com/dhanyajothimani/basic-visualization-and-clustering-in-python
# #heatmap showing the case fatality rate of each country

case_fatality_ratio_heat_map = []
for i in range(len(case_fatality_ratio_per_country)):
	case_fatality_ratio_heat_map.append([case_fatality_ratio_per_country[i]])


sns.heatmap(case_fatality_ratio_heat_map, square=True)
data = dict(type = 'choropleth', 
           	locations = country_list,
           	locationmode = 'country names',
           	z = case_fatality_ratio_per_country, 
           	text = country_list,
           	colorbar = {'title':'Case Fatality Rate'})
layout = dict(title = 'COVID-19 Case Fatality Rate Per Country (Feb 1,2020 - June 30,2020)', geo = dict(showframe = False, projection = {'type': 'mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
plot(choromap3)
# #-------------------------------------------------------------------------------------

# #-------------heatmap showing the total number of confirmed cases in each country (Note: uncomment this section to show heatmap, only 1 heatmap can be generated at a time)----------------------

##Reference: https://www.kaggle.com/dhanyajothimani/basic-visualization-and-clustering-in-python
##heatmap showing the total number of confirmed cases in each country

case_per_country_heat_map = []
for i in range(len(confirmed_cases_per_country)):
	case_per_country_heat_map.append([confirmed_cases_per_country[i]])


sns.heatmap(case_per_country_heat_map, square=True)
data_cases = dict(type = 'choropleth', 
           	locations = country_list,
           	locationmode = 'country names',
           	z = confirmed_cases_per_country, 
           	text = country_list,
           	colorbar = {'title':'Number of Confirmed Cases'})
layout_cases = dict(title = 'COVID-19 Confirmed Cases Per Country (Feb 1,2020 - June 30,2020)', geo = dict(showframe = False, projection = {'type': 'mercator'}))
choromap3_cases = go.Figure(data = [data_cases], layout=layout_cases)
plot(choromap3_cases)
# #-------------------------------------------------------------------------------------


#---------heatmap showing which risk group of each country is labeled (Note: uncomment this section to show heatmap, only 1 heatmap can be generated at a time)------------------------

# #Reference: https://www.kaggle.com/dhanyajothimani/basic-visualization-and-clustering-in-python
# #heatmap showing which risk group of each country is labeled 



risk_labels = []
risk_labels_heat_map = []
for i in range(len(case_fatality_ratio_per_country)):
	risk_labels_heat_map.append([risk_groups[i]])
	risk_labels.append(risk_groups[i])

print(country_list)
print(risk_labels)
print(risk_labels_heat_map)

sns.heatmap(risk_labels_heat_map, square=True)
data_risk_label = dict(type = 'choropleth', 
           	locations = country_list,
           	locationmode = 'country names',
         	z = risk_labels,
           	text = country_list,
           	colorbar = {'title':'2 = Moderate Risk, 0 = Low Risk, 1 = High Risk'})
layout_risk_label = dict(title = 'COVID-19 Risk Level Clusters (Feb 1,2020 - June 30,2020)', geo = dict(showframe = False, projection = {'type': 'mercator'}))
choromap3_risk_label = go.Figure(data = [data_risk_label], layout=layout_risk_label)
plot(choromap3_risk_label)
