# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:19:52 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading files using pandas
def Read_data(file_name):
    df_1=pd.read_csv(file_name)
    df_1=df_1.iloc[[10,41,68,73,89,179,206],[2,5,6,7,8,9,10,11,12,13,14]]
    df_1 = df_1.set_axis(['Country', '1990', '2000', '2012', '2013','2014','2015','2016','2017','2018','2019'], axis=1, inplace=False)
    df_1.reset_index(drop=True, inplace=True)
    df_1 = df_1.dropna()
    print(df_1)
    df_2=df_1.set_index('Country').T
    print(df_2)
    return df_1, df_2


df_1, df11=Read_data("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\A LAND.csv")

print(df11.dtypes)
print(df11.iloc[1])
# to get types of all columns
print(df11.dtypes)
# to get types of individual column
print(df11["Australia"])
print(df11["China"])

# Converting data to numeric type
df_1["1990"] = pd.to_numeric(df_1["1990"])
df_1["2012"] = pd.to_numeric(df_1["2012"])
df_1["2013"] = pd.to_numeric(df_1["2013"])
df_1["2014"] = pd.to_numeric(df_1["2014"])
df_1["2015"] = pd.to_numeric(df_1["2015"])
df_1["2016"] = pd.to_numeric(df_1["2016"])
df_1["2017"] = pd.to_numeric(df_1["2017"])
df11["Australia"] = pd.to_numeric(df11["Australia"])
df11["China"] = pd.to_numeric(df11["China"])
df11["France"] = pd.to_numeric(df11["France"])
df11["Germany"] = pd.to_numeric(df11["Germany"])
df11["India"] = pd.to_numeric(df11["India"])
df11["Spain"] = pd.to_numeric(df11["Spain"])
df11["United States"] = pd.to_numeric(df11["United States"])
print(df11["Australia"])
print(df11["China"])

# ploting datas using pandas
def plot(name,types):
    name.plot(kind=types)
    return
df11.plot()
plot(df11,"hist") 
plot(df_1,"bar") 

# Reading the data forest area
df_forest, df_forest_t=Read_data("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\FOREST.csv")
print(df11.dtypes)
print(df_forest) 
print(df_forest_t)         

# converting data to numeric type
df_forest_t["Australia"] = pd.to_numeric(df_forest_t["Australia"])
df_forest_t["China"] = pd.to_numeric(df_forest_t["China"])
df_forest_t["France"] = pd.to_numeric(df_forest_t["France"])
df_forest_t["Germany"] = pd.to_numeric(df_forest_t["Germany"])
df_forest_t["India"] = pd.to_numeric(df_forest_t["India"])
df_forest_t["Spain"] = pd.to_numeric(df_forest_t["Spain"])
df_forest_t["United States"] = pd.to_numeric(df_forest_t["United States"])   
df_forest["1990"] = pd.to_numeric(df_forest["1990"])
df_forest["2012"] = pd.to_numeric(df_forest["2012"])
df_forest["2013"] = pd.to_numeric(df_forest["2013"])
df_forest["2014"] = pd.to_numeric(df_forest["2014"])
df_forest["2015"] = pd.to_numeric(df_forest["2015"])
df_forest["2016"] = pd.to_numeric(df_forest["2016"])
df_forest["2017"] = pd.to_numeric(df_forest["2017"])                   
   
df_forest_t.plot()                   
plot(df_forest,"bar")  

df_arable, df_arable_t=Read_data("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\Arable Land.csv")
print(df_arable)
print(df_arable_t)

# converting data to numeric type
df_arable_t["Australia"] = pd.to_numeric(df_arable_t["Australia"])
df_arable_t["China"] = pd.to_numeric(df_arable_t["China"])
df_arable_t["France"] = pd.to_numeric(df_arable_t["France"])
df_arable_t["Germany"] = pd.to_numeric(df_arable_t["Germany"])
df_arable_t["India"] = pd.to_numeric(df_arable_t["India"])
df_arable_t["Spain"] = pd.to_numeric(df_arable_t["Spain"])
df_arable_t["United States"] = pd.to_numeric(df_arable_t["United States"])
      
df_arable_t.plot()
plt.xlabel('Country')
plt.ylabel('Area(%)')
plt.title('ARABLE LAND(%)')
plt.savefig('Arableland.png')
plt.show()

plot(df_arable,"bar")  
plt.xlabel('Country')
plt.ylabel('Area(%)')
plt.title('ARABLE LAND(%)')
plt.savefig('Arableland.png')
plt.show()


# reading the data total population
df_population, df_population_t=Read_data("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\TOTAL POPULATION.csv")
print(df_population)
print(df_population_t)
df_population_t["Australia"] = pd.to_numeric(df_population_t["Australia"])
df_population_t["China"] = pd.to_numeric(df_population_t["China"])
df_population_t["France"] = pd.to_numeric(df_population_t["France"])
df_population_t["Germany"] = pd.to_numeric(df_population_t["Germany"])
df_population_t["India"] = pd.to_numeric(df_population_t["India"])
df_population_t["Spain"] = pd.to_numeric(df_population_t["Spain"])
df_population_t["United States"] = pd.to_numeric(df_population_t["United States"])

# line graph for total population
df_population_t.plot()
plt.xlabel('Year')
plt.ylabel('population(million)')
plt.title('TOTAL POPULATION (MILLION)')
plt.savefig('Totalpopulation.png')
plt.show()

# reading the file population growth
df_population= pd.read_csv("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\P GROWTH.csv")
print(df_population)
# analysing Australias population growth from 2015 to 2021 using piechart
Year = ['2015', '2016', '2017', '2018', '2019', '2020', '2021']
Population = np.array([1.439216653, 1.561940498, 1.684523108, 1.536105516,1.521653677, 1.282933, 0.178832418])
 
plt.figure()
plt.pie(Population, labels=Year)
plt.title('POPULATION GROWTH (AUSTRALIA)', size=8)
plt.savefig('pgrowth.png')
plt.show()

aver = np.mean(df_agriculture_t["India"])
print("Average agriculture land area of India:", aver)
std = np.std(df_agriculture_t["India"])
print("Std. deviation agricultural land area of India:", std)
print("Skew:", stats.skew(df_agriculture_t["India"]))
print("Kurtosis", stats.kurtosis(df_agriculture_t["India"]))



    
