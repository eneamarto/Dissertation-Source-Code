import geopandas as gpd
import osmnx as ox
import itertools
from difflib import SequenceMatcher

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from pylab import rcParams

import collections
from statistics import mean 

import networkx as nx

import matplotlib.cm as cm
import matplotlib.colors as colors

ox.config(log_console=True, use_cache=True)

def similar(a, b):
    if (SequenceMatcher(None, a, b).ratio() >=0.77):
        return True
    else:
        return False
    

# Create connection to SqLite
connection = sqlite3.connect("C:\\Users\\Enea\\Desktop\\Senior Project\\sp.db")

cur = connection.cursor()

detection_df = pd.read_sql_query("SELECT * from detections where timestamp LIKE '2020-12-28%' and timestamp not like '2020-12-28_06%'", connection)

connection.close()

#Group DataFrame by road_name

df_road_names = detection_df.groupby('road_name').apply(lambda x: [(list(x['timestamp']), list(x['car_count']))]).apply(pd.Series)

## Road Charts


rcParams['figure.figsize'] = 25, 20

for i in range(70):

    ts_size = sorted(list(set((df_road_names[0][i][0][:]))))
    
    ct_size = df_road_names[0][i][1][:len(ts_size)]
    
    fig, ax = plt.subplots()

    ax.grid(True)

    fig.autofmt_xdate()

    ax.xaxis.set_major_locator(plt.MaxNLocator(2))

    plt.locator_params(axis='x', nbins=70)

    ax.set_title(df_road_names.index[i], fontdict={'fontsize': 25, 'fontweight': 'medium'})
    
    ax.plot(ts_size,ct_size)

    plt.show()
    

## Live Map


file_path = "C:\\Users\\Enea\\Desktop\\Senior Project\\camera_ip_v2.txt"

road_names =[]
road_ips = []

road_dict = {}

with open(file_path,'r', encoding='utf-8') as file:

    for line1,line2 in itertools.zip_longest(*[file]*2):
        road_dict[line1.rstrip('\n')] = line2.rstrip('\n')


# Match road_ids of camera_ips and Dataframe

road_df = []

for i, row in detection_df[['road_name','road_id']].iterrows():
    for ip in road_dict.keys():
        if row['road_id'] == ip:
            road_df.append(road_dict[ip])
            
            
# create dataframe with the above list 

rdf = pd.DataFrame(road_df)
rdf.columns = ['road_name']

# Replace road_name values of the Dataframe with the correct road_names from camera_ips

detection_df['road_name'] = rdf['road_name']


# remove seconds and single digit minutes from timestamp
# leaving only increments of 10 minutes on the timestamps 

df_time_rounded = detection_df.replace(to_replace=r'\d{1}[-]\d{2}$',regex=True,value='0-00')


#aggregate the data as to be grouped by timestamp and take the average of each road meassurement

dk = df_time_rounded.groupby('timestamp').apply(lambda x: [(list(zip(x['road_name'], x['car_count'])))]).apply(pd.Series)
dk.columns=["Name-Count"]

dict_of_dictionaries = {}

for i in range(len(dk["Name-Count"])-1):
    c = collections.defaultdict(list)

    for a,b in dk["Name-Count"][i]:
        c[a].append(b)


    for key in c.keys():
        value_list = c[key]

        c[key] = round(mean(value_list),2)

    dict_of_dictionaries[dk.index[i]] = c


#final dataframe with the average car_count for each road grouped by 10 min intervals. 

df_aggregate=pd.DataFrame.from_dict(dict_of_dictionaries,orient='columns').transpose()

## Group by road_name to find the quantiles for each road

df_quantiles = df_time_rounded.groupby('road_name').apply(lambda x: [list(x['car_count'])]).apply(pd.Series)
df_quantiles.columns = ["car_count"]

## calculate quantiles for each road

def interval_to_list(interval):
    l = []
    
    for i in interval:
        l.append(tuple(i))
        
    return l

road_quantiles = {}

for index,values in df_quantiles['car_count'].iteritems():
     
    road_quantiles[index] = interval_to_list(pd.qcut(values,3,duplicates='drop').categories.to_tuples())


diff = []

for index,data in df_aggregate.iterrows():
    diff.append(data.index)
l = diff[0].to_list()


cf = '["highway"~"motorway|primary|secondary|tertiary"]'

G= ox.graph_from_bbox(41.3624,41.3035,19.7541,19.8889, network_type='drive',custom_filter=cf,simplify=True)

pg = ox.graph_to_gdfs(G,nodes=False).fillna('')

edge_names = list(pg.name)

def check_tuple(tupl_list,val):
    
    position= 0
    
    for i in range(len(tupl_list)):
            
        if tupl_list[i][0] < val < tupl_list[i][1]:
            
            position = i
            
    if position is 0:
        return 'g'
    elif position is 1:
        return 'y'
    elif position is 2:
        return 'r'

def find_pos(name):
    pos = [] 
    
    for i in range(len(edge_names)):
        if edge_names[i] == name:
            pos.append(i)
    return pos


measure= dict(zip(df_aggregate.loc['2020-12-28_09-10-00'].index, df_aggregate.loc['2020-12-28_09-10-00'].values))

ed = ['w']*1157

for u, v, key, data in G.edges(keys=True, data=True):   

    name = data.get('name')
    
    if name != None :
        
        if name in  measure.keys():
            
            color = check_tuple(road_quantiles[name],measure[name])
            
            for i in find_pos(name):
                
                ed[i] = color
                
        
fig, ax = ox.plot_graph(G, node_color='b', node_edgecolor='b', node_size=30,
                        edge_color=ed, edge_linewidth=5,figsize=(40,40))



ec = ['g' if data.get('name') in l  else 'w'  for u, v, key, data in G.edges(keys=True, data=True)]

fig, ax = ox.plot_graph(G, node_color='b', node_edgecolor='b', node_size=30,
                        edge_color=ec, edge_linewidth=5,figsize=(40,40))



edge_centrality = nx.closeness_centrality(nx.line_graph(G))

ev = [edge_centrality[edge + (0,)] for edge in G.edges()]


norm = colors.Normalize(vmin=min(ev)*0.8, vmax=max(ev))
cmap = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
ec = [cmap.to_rgba(cl) for cl in ev]

fig, ax = ox.plot_graph(G, bgcolor='k',  node_size=25, node_color='w', node_edgecolor='gray', node_zorder=2,
                        edge_color=ec, edge_linewidth=5, edge_alpha=1,figsize=(40,40))
