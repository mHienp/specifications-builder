# YOUR INPUTS HERE

tree = 'Douglas-fir - Genus type'
age = [10,30,50,70,90,110,130,150,170,190,210]
biomass = [52.34, 179.87, 243.72, 240.58, 239.97, 268.77, 275.11, 346.75, 410.24, 365.98, 496.13]
initial_guess = [9215.46768485293, 0.0000212284925046361, 0.565657070848123]

# import libraries

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rst
from rasterio.plot import show, show_hist
from rasterio.mask import mask

import matplotlib as mpl
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# function fit
from scipy.optimize import curve_fit

# save output
import csv

# Richard-Chapman
def growth_curve(age, A, k, p):
    return A * (1 - np.exp(-k * age))**p

# age
x_data_hs = np.array(age)
# AGB
y_data_hs = np.array(biomass)

params_hs, _ = curve_fit(growth_curve, x_data_hs, y_data_hs, initial_guess)
params_hs

plt.scatter(x_data_hs, y_data_hs, label='data', marker='x', c='r')
x = np.linspace(1, 210, 210)
plt.plot(x, growth_curve(x, *params_hs), 'b-', label='growth curve')
plt.title('GROWTH CURVE')
plt.xlabel('Age (years)')
plt.ylabel('AGB (Mg ha-1)')
plt.legend()
plt.savefig('growth.png', dpi=300)
# plt.show()

# load forest shape
shp = 'boundaries/forest.shp'
forest = gpd.read_file(shp)

# plot
cmap='Set3'
fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.set_title('ECOZONES')
forest.plot(ax=ax,column='ECO_NAME',cmap=cmap,edgecolor='black',legend=True,legend_kwds= {'ncol': 1, 'loc': 'lower left'})
# plt.show()

ax.figure.savefig('ecozones.png', dpi=300)

# load the Holdridge Lize Zones dataset
shp = 'datasets/HoldridgeLifeZones.json'
holdridge_df = gpd.read_file(shp)

# clip to forest boundaries
forest_holdrige_df = holdridge_df.clip(forest)
forest_holdrige_df.shape

# replace or delete empty rows
forest_holdrige_df.DESC = forest_holdrige_df.DESC.replace(r'^\s*$', 'Unspecified', regex=True)
forest_holdrige_df=forest_holdrige_df[forest_holdrige_df['DESC']!='Unspecified']
forest_holdrige_df.shape

# rename and change columns
forest_holdrige_df.rename(columns={'DESC':'LifeZone'}, inplace=True)
forest_holdrige_df.loc[:,'SYMBOL'] = 0
forest_holdrige_df.rename(columns={'SYMBOL':'Age'}, inplace=True)

# plot
cmap='tab20_r'
fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.set_title('HOLDRIDGE LIFEZONES')
forest_holdrige_df.plot(ax=ax,column='LifeZone',cmap=cmap,edgecolor='black',legend=True,legend_kwds= {'ncol': 1, 'loc': 'lower left'})
# plt.show()

ax.figure.savefig('holdridge.png', dpi=300)

# artifact
forest_holdrige_df.to_file("inventory.shp")

# open tif file from folder
temp_rst = 'datasets/wc2.1_2.5m_bio_1.tif'
temp_img = rst.open(temp_rst)

# clip raster
out_temp, out_transform = mask(temp_img, [geom for geom in forest.geometry], crop=True, pad=True)
out_meta = temp_img.meta

out_meta.update({"driver": "GTiff",
                 "height": out_temp.shape[1],
                 "width": out_temp.shape[2],
                 "transform": out_transform})

with rst.open(r'Forest-BioClim2.5m_Temperature.tiff', 'w', **out_meta) as dest:
    dest.write(out_temp)

# load raster
forest_temp_rst = r'Forest-BioClim2.5m_Temperature.tiff'
forest_temp_img = rst.open(forest_temp_rst)

mpl.rc('image', cmap='gist_earth')

arr = forest_temp_img.read(1)
arr = np.where(arr < -20, np.nan, arr)

# plot
fig, ax = plt.subplots(1, 1, figsize=(15,10))

ax.set_title('ANNUAL MEAN TEMPERATURE')
show(arr, transform=forest_temp_img.transform, ax=ax)
forest.plot(ax=ax,color='none',edgecolor='black',legend=True)

fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))), ax=ax)

# plt.show()

ax.figure.savefig('temperature.png', dpi=300)

agb_values = [growth_curve(age, *params_hs) for age in range(201)]
header     = ['LifeZone', 'AIDBSPP'] + list(range(201))
life_zones = sorted(forest_holdrige_df['LifeZone'].unique())
species    = tree

with open('Growth_Curves.csv', 'w') as gc_file:
    gc_writer = csv.writer(gc_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    gc_writer.writerow(header)
    for life_zone in life_zones:
        gc_writer.writerow([life_zone] + [species] + agb_values)
        
# group by lifezones
groups = forest_holdrige_df.groupby('LifeZone')

# create empty lists to later create a dataframe
zones = []
max_temps = []
min_temps = []

# Loop through each group and create a geodataframe for that group
for name, group in groups:

    zones.append(name)  
    
    # Replace spaces with underscores in the name
    name_underscore = name.lower().replace(' ', '_')

    # Create the variable name
    var_name = name_underscore

    # Create a new GeoDataFrame for the group
    globals()[var_name] = gpd.GeoDataFrame(group)

    # clip raster
    out_temp, out_transform = mask(temp_img, [geom for geom in globals()[var_name].geometry], crop=True, pad=True)
    out_meta = temp_img.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_temp.shape[1],
                     "width": out_temp.shape[2],
                     "transform": out_transform})

    with rst.open(r'zone.tiff', 'w', **out_meta) as dest:
        dest.write(out_temp)

    # Open the raster dataset
    with rst.open('zone.tiff') as dataset:
        # Read the entire raster dataset into a NumPy array
        raster_array = dataset.read()

        # Get the maximum value of the raster dataset
        max_val = round(raster_array.max(), 2)
        max_temps.append(max_val)

        # Flatten the NumPy array into a 1D array
        flat_array = raster_array.flatten()

        # Get the unique values in the 1D array
        unique_values = np.unique(flat_array)

        # Get the second minimum value
        second_min_val = round(unique_values[1], 2)
        min_temps.append(second_min_val) 
        
# create the dataframe
holdridge_temp_df = pd.DataFrame({'LifeZone': zones, 'Low (°C)': min_temps, 'High (°C)': max_temps})

# set the display options to show only 2 decimal places
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

# set the 'Name' column as the index
holdridge_temp_df.set_index('LifeZone', inplace=True)

# display the dataframe in markdown format
# print(holdridge_temp_df.to_markdown())

fig, axhist = plt.subplots(1, 1)
show_hist(forest_temp_img,bins=5,ax=axhist)
axhist.get_legend().remove()
axhist.axes.get_yaxis().set_visible(False)
axhist.set_xlabel('°C')
axhist.set_title('ANNUAL MEAN TEMPERATURE DISTRIBUTION')

axhist.figure.savefig('histogram.png', dpi=300)

# texts in report

eco_zones=sorted(forest['ECO_NAME'].unique())
eco_zones_str = "\n* ".join(eco_zones)
eco_string = "The econames of the forests' sub-areas are:\n* {}\n### Holdridge Lifezones".format(eco_zones_str)

with open("econames.txt", 'w') as outfile:
        outfile.write(eco_string)

# life_zones_str = "\n* ".join(life_zones)
holdridge_str = 'The Holdrige lifezones of the forests, as well as their ranges of temperatures are:\n' + holdridge_temp_df.to_markdown() + '\n\nThe distribution of the annual mean temperatures across the area is as followed:'

with open("holdridge.txt", 'w') as outfile:
        outfile.write(holdridge_str)

growth_str = "### Growth curve to be used in the simulation\n{} is the leading tree species. The curve below shows the growth in tree biomass through age.".format(tree)
with open("tree.txt", 'w') as outfile:
        outfile.write(growth_str)
