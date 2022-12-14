# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:19:52 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib   # used to do the formatting
import matplotlib.pyplot as plt
import scipy.stats as stats  # good module for statistical operations

#  descibing function to Read files using pandas
def Read_data(file_name):
    df_1=pd.read_csv(file_name)
    # slicing the data
    df_1=df_1.iloc[[10,41,68,73,89,179,206],[2,5,6,7,8,9,10,11,12,13,14]]
    df_1 = df_1.set_axis(['Country', '1990', '2000', '2012', '2013','2014','2015','2016','2017','2018','2019'], 
                         axis=1, inplace=False)
    df_1.reset_index(drop=True, inplace=True)
    df_1 = df_1.dropna()
    df_1=df_1.set_index('Country')
    print(df_1)
    # transposing the data
    df_2=df_1.T
    print(df_2)
    return df_1, df_2

#  describing function to plot datas using pandas
def plot(name,types):
    name.plot(kind=types)
    return

# function to plot heatmap
def heatmap(data, row_labels, column_labels):
    im = ax.imshow(values)
    # colorbar in the heatmap
    cbar = ax.figure.colorbar(im)
    ax.set_xticks(np.arange(data.shape[1]), labels=column_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # adding separation lines between the values
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im,cbar

# function to annotate heatmap
def heatmap_2(im, data=None, valfmt="{x:.2f}",textcolors=("black", "white"),
              threshold=None):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
        if threshold is not None:
            threshold=im.norm(threshold)
        else:
            threshold=im.norm(data.max())/2
            # arrangement of the values in each column
        arrange=dict(horizontalalignment="center", verticalalignment="center")
        if isinstance(valfmt, str):
            # formatting using matplotlib
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
            texts = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    arrange.update(color=textcolors[int(im.norm(data[i,j]) > threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None),arrange)
                    texts.append(text)
            return texts
        
# Reading files 
df_agriculture, df_agriculture_t=Read_data("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\A LAND.csv")
print(df_agriculture)
print(df_agriculture_t)

print(df_agriculture.describe())

print(df_agriculture_t.dtypes)
print(df_agriculture.iloc[:,0:11])
# to get types of all columns
print(df_agriculture_t.dtypes)
# to get types of individual column
print(df_agriculture_t["Australia"])
print(df_agriculture_t["China"])

# Converting data to numeric type
df_agriculture["1990"] = pd.to_numeric(df_agriculture["1990"])
df_agriculture["2012"] = pd.to_numeric(df_agriculture["2012"])
df_agriculture["2013"] = pd.to_numeric(df_agriculture["2013"])
df_agriculture["2014"] = pd.to_numeric(df_agriculture["2014"])
df_agriculture["2015"] = pd.to_numeric(df_agriculture["2015"])
df_agriculture["2016"] = pd.to_numeric(df_agriculture["2016"])
df_agriculture["2017"] = pd.to_numeric(df_agriculture["2017"])
df_agriculture_t["Australia"] = pd.to_numeric(df_agriculture_t["Australia"])
df_agriculture_t["China"] = pd.to_numeric(df_agriculture_t["China"])
df_agriculture_t["France"] = pd.to_numeric(df_agriculture_t["France"])
df_agriculture_t["Germany"] = pd.to_numeric(df_agriculture_t["Germany"])
df_agriculture_t["India"] = pd.to_numeric(df_agriculture_t["India"])
df_agriculture_t["Spain"] = pd.to_numeric(df_agriculture_t["Spain"])
df_agriculture_t["United States"] = pd.to_numeric(df_agriculture_t["United States"])
print(df_agriculture_t["Australia"])
print(df_agriculture_t["China"])


plot(df_agriculture,"bar") 
plt.xlabel('Country')
plt.ylabel('Area(%)')
plt.title('AGICULTURAL LAND(%)')
plt.savefig('Agriculturalland.png')
plt.show()

df_agriculture_t.plot()
plt.xlabel('Year')
plt.ylabel('Area(%)')
plt.title('AGICULTURAL LAND(%)')
plt.savefig('Agriculturalland.png')
plt.show()


# Reading the data forest area
df_forest, df_forest_t=Read_data("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\FOREST.csv")
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
plt.xlabel('Country')
plt.ylabel('Area(%)')
plt.title('FOREST AREA(%)')
plt.savefig('Forestarea.png')
plt.show()

# reading the data arable land area
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
df_arable["1990"] = pd.to_numeric(df_arable["1990"])
df_arable["2012"] = pd.to_numeric(df_arable["2012"])
df_arable["2013"] = pd.to_numeric(df_arable["2013"])
df_arable["2014"] = pd.to_numeric(df_arable["2014"])
df_arable["2015"] = pd.to_numeric(df_arable["2015"])
df_arable["2016"] = pd.to_numeric(df_arable["2016"])
df_arable["2017"] = pd.to_numeric(df_arable["2017"])
      
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
Populationgrowth = np.array([1.439216653, 1.561940498, 1.684523108, 1.536105516,1.521653677, 1.282933, 0.178832418])
 
plt.figure()
plt.pie(Populationgrowth, labels=Year)
plt.title('POPULATION GROWTH (AUSTRALIA)', size=8)
plt.savefig('pgrowth.png')
plt.show()

df_poverty= pd.read_csv("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\poverty head count.csv")
print(df_poverty)
df_poverty = df_poverty.set_axis(['Country', '1990', '2000', '2012', '2013','2014','2015','2016','2017','2018','2019'], 
                                 axis=1, inplace=False)
df_poverty.reset_index(drop=True, inplace=True)
df_poverty1 = df_poverty.dropna()
print(df_poverty1)
df_poverty2=df_poverty1.set_index('Country')
print(df_poverty2)


plot(df_poverty2,"bar") 
plt.xlabel('Country')
plt.ylabel('poverty head count (% of population)')
plt.title('POVERTY HEAD COUNT (% OF POPULATION)')
plt.savefig('Pvertyhead.png')
plt.show()

# ploting heatmap of different climate change factors of india
India = pd.read_csv("C:\\Users\\DELL\\Downloads\\AD ASSIGNMENT2\\Indiadata.csv")
print(India)
factors = ["Forest land", "Agriculture land", "Arable land", "Population(billion)",
                  "GDP(%)", "Greenhouse emission(millionton)", "Urban population(%)"]
years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018"]

# creating array from the "India" data to plot the heatmap
values = np.array([[23.6, 23.7, 23.8, 23.9 , 24.1, 24.0, 24.1],
                        [60.4, 60.4,  60.4, 60.3, 60.2, 60.2, 60.2],
                        [52.6,  52.6, 52.6, 52.5, 52.4, 52.2, 52.2],
                        [1.2, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3],
                        [16.8, 17.1, 16.8, 16.2, 16.4, 16.6, 16.0],
                        [2.7, 2.8, 3.0, 3.0, 3.1, 3.2, 3.3],
                        [31.6, 32.0, 32.4, 32.8, 33.2, 33.6, 34.0]])
   


fig, ax = plt.subplots()
im, cbar = heatmap(values, factors, years)
texts = heatmap_2(im, valfmt="{x:.1f} ")

fig.tight_layout()
plt.title("INDIA")
plt.show()

# analysing the statistical properties of india's agricultural land area
# calculating average
aver = np.mean(df_agriculture_t["India"])
print("Average agriculture land area of India:", aver)

# calculating standared deviation
std = np.std(df_agriculture_t["India"])
print("Std. deviation agricultural land area of India:", std)

# calculating skewness
print("Skew:", stats.skew(df_agriculture_t["India"]))

# calculating kurtosis
print("Kurtosis:", stats.kurtosis(df_agriculture_t["India"]))

# calculating median
median = df_agriculture_t['India'].median()
print(median)

    
