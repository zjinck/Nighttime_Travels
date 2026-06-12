# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:35:56 2026

@author: Zhihua
"""
import os
import time
import timeit
import csv
import numpy as np
import math
import pandas as pd
import array as arr
import sys
import glob
import zipfile
from fastkml import kml
import datetime
from datetime import datetime, date
#geo
import geopandas as gpd
from geopandas import GeoDataFrame
import folium
from folium.plugins import MarkerCluster
from folium import FeatureGroup,LayerControl,Choropleth
import seaborn as sns
import contextily as ctx
import branca.colormap as cm
from branca.colormap import linear
import geopy.distance
from shapely.ops import unary_union
from shapely import wkt
from shapely.ops import nearest_points
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
import json
#maths & figures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm, TwoSlopeNorm
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import branca.colormap as cm
import plotly.express as px
import plotly.graph_objects as go
import statistics
import itertools
from math import radians, sin, cos, sqrt, atan2
import heapq
import cartopy.crs as ccrs
import mpu
import statsmodels.api as sm
import plotly.graph_objects as go
#%% survey data
# read data, recode mode
pd.set_option('display.max_columns', None)
pd.reset_option('display.float_format')
trip_unlinked = pd.read_csv(r"C:\Users\Zhihua\Documents\MyResearch\Chicago_KY\submission\TravelSurveys_PublicData_CMAP\MyDailyTravel_2024_2025Phase1\MyDailyTravelData\trip_unlinked.csv")
mode_recode = {1:'walk',2:'micro',4:	'micro',5:	'taxi/TNC',6:	'taxi/TNC',7:'other',
8:'auto',9:'taxi/TNC',10:'transit',11:'taxi/TNC',12:'other',13:'transit',
14:'other',995:'	Missing Response'}
trip_unlinked['mainmode'] = (trip_unlinked['mode_type'].map(mode_recode))
dfchi23 = trip_unlinked
dfchi19 = pd.read_csv(r"C:\Users\Zhihua\Documents\MyResearch\Chicago_KY\submission\TravelSurveys_PublicData_CMAP\MyDailyTravel_2018_2019\MyDailyTravelData\place.csv")
dfchi19loc = pd.read_csv(r"C:\Users\Zhihua\Documents\MyResearch\Chicago_KY\submission\TravelSurveys_PublicData_CMAP\MyDailyTravel_2018_2019\MyDailyTravelData\location.csv")
dfchi19home = dfchi19loc[dfchi19loc['home']==1]
dfchi19home['homeloc'] = dfchi19home.locno
dfchi19 = dfchi19.merge(dfchi19home[['homeloc', 'sampno']], on='sampno',  how='left')
dfchi19 = dfchi19.sort_values(['sampno', 'perno', 'traveldayno', 'placeno'])
group_cols = ['sampno', 'perno', 'traveldayno']
mode_recode = {
101:	'walk',
102:	'micro',
103:	'micro',
104:	'micro',
201:	'micro',
202: 'auto',
203: 'auto',
301: 'taxi/TNC',
401:'transit',
500: 'transit',
501: 'transit',
502: 'transit',
503: 'transit',
504: 'transit',
505: 'transit',
506: 'transit',
509: 'transit',
601: 'taxi/TNC',
701: 'taxi/TNC',
702: 'auto',
703: 'auto',
704: 'taxi/TNC',
705: 'taxi/TNC',
801: 'other',}
dfchi19['mainmode'] = (dfchi19['mode'].map(mode_recode))
# purpose recode
dfchi19['tpur_prev'] = dfchi19.groupby(group_cols)['tpurp'].shift(1)
dfchi19['tpur_next'] = dfchi19.groupby(group_cols)['tpurp'].shift(-1)
dfchi19['placeno_prev'] = dfchi19.groupby(group_cols)['placeno'].shift(1)
dfchi19['placeno_next'] = dfchi19.groupby(group_cols)['placeno'].shift(-1)
dfchi19['pur_prior'] = None
dfchi19['loc_prior'] = None
mask_prev = dfchi19['placeno'] > 1
dfchi19.loc[mask_prev, 'pur_prior'] = dfchi19.loc[mask_prev, 'tpur_prev']
mask_first = dfchi19['placeno'] == 1
dfchi19.loc[mask_first, 'pur_prior'] = ( (dfchi19.loc[mask_first, 'locno'] == dfchi19.loc[mask_first, 'homeloc']).astype(int))
dfchi19['pur_later'] = None
mask_next = dfchi19['placeno_next'].notna()
dfchi19.loc[mask_next, 'pur_later'] = dfchi19.loc[mask_next, 'tpur_next']
mask_last = ~mask_next
dfchi19.loc[mask_last, 'pur_later'] = ( np.where(dfchi19.loc[mask_last, 'locno'] == dfchi19.loc[mask_last, 'homeloc'],   1,  -8))
valid_prev = dfchi19['placeno'] - dfchi19['placeno_prev'] == 1
dfchi19.loc[mask_prev & valid_prev, 'pur_prior'] = dfchi19.loc[mask_prev & valid_prev, 'tpur_prev']
conds = [(dfchi19['pur_prior'].isin([1,2])),(~dfchi19['pur_prior'].isin([1,2])) & (dfchi19['tpurp'].isin([1, 2]))]
dfchi19['basepur'] = np.select(conds, ['HB', 'HB'], default='NHB')
activity_map = {
    1: 'O',
    2: 'W', 3: 'W', 4: 'W', 5: 'W',
    6: 'E',
    7: 'O',
    8: 'S', 9: 'S', 10: 'S', 11: 'S',
    12: 'O', 13: 'O', 14: 'O',
    15: 'S',
    16: 'R', 17: 'R', 18: 'R', 19: 'R', 20: 'R', 21: 'R',
    22: 'R', 23: 'R', 24: 'R', 25: 'R',
    26: 'O', 27: 'O',
    28: 'O', 97: 'O'
}
dfchi19['mapped_tpurp'] = dfchi19['tpurp'].map(activity_map).fillna('O')
dfchi19['mapped_ppurp'] = dfchi19['pur_prior'].map(activity_map).fillna('O')
is_hb1 = dfchi19['pur_prior'].isin([1, 2])
is_hb2 = (~is_hb1) & dfchi19['tpurp'].isin([1, 2])
is_nhb = (~is_hb1) & (~dfchi19['tpurp'].isin([1, 2]))
conds = [is_hb1, is_hb2, is_nhb]
choices = ['HB' + dfchi19['mapped_tpurp'], 'HB' + dfchi19['mapped_ppurp'],'NHB'+dfchi19['mapped_ppurp']]
dfchi19['trippur'] = np.select(conds, choices, default='NHBO')
dfchi19['trippur'] = dfchi19['trippur'].replace('NHBE', 'NHBO')
dfchi19['trippur'] = dfchi19['trippur'].replace('NHBS', 'NHBO')
dfchi19['trippur'] = dfchi19['trippur'].replace('NHBR', 'NHBO')
owesr_map = {
    1: 'H',
    2: 'W', 3: 'W',
    4: 'E', 5: 'E',
    6: 'O',
    7: 'S',
    8: 'R', 9: 'R',
    10: 'S',
    11: 'O', 12: 'O', 13: 'O', -1: 'O'
}
dfchi23['o_owesr'] = dfchi23['o_purpose_category'].map(owesr_map).fillna('O')
dfchi23['d_owesr'] = dfchi23['d_purpose_category'].map(owesr_map).fillna('O')
dfchi23['base'] = np.where((dfchi23['o_owesr']=='H')|(dfchi23['d_owesr']=='H'), 'HB', 'NHB')
dfchi23['activity'] = np.where(dfchi23['d_owesr']=='H', dfchi23['o_owesr'], dfchi23['d_owesr'])
dfchi23['trippur'] = dfchi23['base'] + dfchi23['activity']
dfchi23['trippur'] = dfchi23['trippur'].replace('NHBE', 'NHBO')
dfchi23=dfchi23[(dfchi23.o_in_region==1)&(dfchi23.d_in_region==1)]
dfchi19per = pd.read_csv(r"C:\Users\Zhihua\Documents\MyResearch\Chicago_KY\submission\TravelSurveys_PublicData_CMAP\MyDailyTravel_2018_2019\MyDailyTravelData\person.csv")
dfchi19['depart_hour'] =pd.to_datetime(dfchi19.deptime).dt.hour
dfchi23['tweight'] = dfchi23.trip_weight #linked_trip_weight
dfchi19 = dfchi19.merge(dfchi19per[['wtperfin','perno','sampno']], on = ['perno','sampno'], how= 'left')
dfchi19['tweight'] = dfchi19.wtperfin
#%% baseic bar plots
# mode
chint23 = dfchi23[(dfchi23.depart_hour>=20)|(dfchi23.depart_hour<6)]
chidy23 = dfchi23[(dfchi23.depart_hour<20)&(dfchi23.depart_hour>=6)]
chint19 = dfchi19[(dfchi19.depart_hour>=20)|(dfchi19.depart_hour<6)]
chidy19 = dfchi19[(dfchi19.depart_hour<20)&(dfchi19.depart_hour>=6)]
chi19_pct={k:v/sum([chint19["tweight"].sum(),chidy19["tweight"].sum()])*100 for k,v in {"Night":chint19["tweight"].sum(),"Day":chidy19["tweight"].sum()}.items()}
chi23_pct={k:v/sum([chint23["tweight"].sum(),chidy23["tweight"].sum()])*100 for k,v in {"Night":chint23["tweight"].sum(),"Day":chidy23["tweight"].sum()}.items()}

citievalues={"2019":(chint19,chidy19,"tweight"),"2023":(chint23,chidy23,"tweight")}
order=["walk","transit","micro","taxi/TNC","auto"]
res = {}
colors=dict(zip(order,plt.cm.tab10.colors[:len(order)]))
for city, (nt, dy, wcol) in citievalues.items():
    if wcol and wcol in nt.columns and wcol in dy.columns:
        ntv = nt.assign(w=pd.to_numeric(nt[wcol], errors="coerce")).groupby("mainmode")["w"].sum()
        dyv = dy.assign(w=pd.to_numeric(dy[wcol], errors="coerce")).groupby("mainmode")["w"].sum()
    else:
        ntv = nt["mainmode"].value_counts()
        dyv = dy["mainmode"].value_counts()
    res[(city, "Night")] = ntv.reindex(order).fillna(0) / ntv[ntv.index.isin(order)].sum() * 100
    res[(city, "Day")] = dyv.reindex(order).fillna(0) / dyv[dyv.index.isin(order)].sum() * 100
df = pd.DataFrame(res).T
cities = list(citievalues.keys())
x = np.arange(len(cities))
w = 0.45
fig, ax = plt.subplots(figsize=(2.9,3.5))#(4.5,3.5))
for i, t in enumerate(["Night", "Day"]):
    bottoms = np.zeros(len(cities))
    for p in order:
        vals = [df.loc[(c, t), p] for c in cities]
        bars = ax.bar(x + (i-0.5)*w, vals, w, bottom=bottoms, color=colors[p],
                      label=p if i == 0 else "")
        for j, (b, v) in enumerate(zip(bars, vals)):
            if v > 3: 
                ax.text(b.get_x() + b.get_width()/2, bottoms[j] + v/2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9)
        bottoms += vals
ax.set_xticks(x);ax.set_xticklabels(cities);ax.set_ylabel("Mode")
ax2 = ax.twiny();ax2.set_xlim(ax.get_xlim());ax2.set_xticks(x)
ax2.set_xticklabels(["Night / Day"]*len(cities))
for xi in x:
    ax.axvline(x=xi, color='gray', linestyle='--', alpha=0.4)
plt.tight_layout();plt.savefig("chicompmode.png");plt.show()
# purpose
order=["HBW","HBE","NHBW","HBS","HBR","HBO","NHBO"]
res = {}
colors=dict(zip(order,plt.cm.tab10.colors[:len(order)]))

for city, (nt, dy, wcol) in citievalues.items():
    if wcol and wcol in nt.columns and wcol in dy.columns:
        ntv = nt.assign(w=pd.to_numeric(nt[wcol], errors="coerce")).groupby("trippur")["w"].sum()
        dyv = dy.assign(w=pd.to_numeric(dy[wcol], errors="coerce")).groupby("trippur")["w"].sum()
    else:
        ntv = nt["trippur"].value_counts()
        dyv = dy["trippur"].value_counts()
    res[(city, "Night")] = ntv.reindex(order).fillna(0) / ntv[ntv.index.isin(order)].sum() * 100
    res[(city, "Day")] = dyv.reindex(order).fillna(0) / dyv[dyv.index.isin(order)].sum() * 100
df = pd.DataFrame(res).T
cities = list(citievalues.keys())
x = np.arange(len(cities))
w = 0.45
fig, ax = plt.subplots(figsize=(2.9,3.5))#(4.2,3.5))
for i, t in enumerate(["Night", "Day"]):
    bottoms = np.zeros(len(cities))
    for p in order:
        vals = [df.loc[(c, t), p] for c in cities]
        bars = ax.bar(x + (i-0.5)*w, vals, w, bottom=bottoms, color=colors[p],
                      label=p if i == 0 else "")
        for j, (b, v) in enumerate(zip(bars, vals)):
            if v > 3: 
                ax.text(b.get_x() + b.get_width()/2, bottoms[j] + v/2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9)
        bottoms += vals
ax.set_xticks(x);ax.set_xticklabels(cities);ax.set_ylabel("Purpose")
ax2 = ax.twiny();ax2.set_xlim(ax.get_xlim());ax2.set_xticks(x)
ax2.set_xticklabels(["Night / Day"]*len(cities))
for xi in x:
    ax.axvline(x=xi, color='gray', linestyle='--', alpha=0.4)
plt.tight_layout();plt.savefig("chicomppur.png");plt.show()
# counts/weights/percent
for year, (nt_df, dy_df, wcol) in citievalues.items():
    nt_df[wcol] = pd.to_numeric(nt_df[wcol], errors="coerce")
    dy_df[wcol] = pd.to_numeric(dy_df[wcol], errors="coerce")
labels = []; nt_vals = []; dy_vals = []
for year, (nt_df, dy_df, wcol) in citievalues.items():
    labels.append(year)
    nt_vals.append(nt_df[wcol].sum(skipna=True))
    dy_vals.append(dy_df[wcol].sum(skipna=True))
nt_vals = np.array(nt_vals, dtype=float)
dy_vals = np.array(dy_vals, dtype=float)
totals = nt_vals + dy_vals
x = np.arange(len(labels))
plt.figure(figsize=(2.9,2.7))
plt.bar(x, nt_vals, color="tab:blue", label="NT",alpha =0.75)
plt.bar(x, dy_vals, bottom=nt_vals, color="tab:orange", label="DY",alpha =0.75)
for i in range(len(labels)):
    nt_pct = 100 * nt_vals[i] / totals[i]
    dy_pct = 100 * dy_vals[i] / totals[i]
    plt.text(x[i], nt_vals[i]/2, f"{nt_vals[i]:,.0f}\n{nt_pct:.1f}%", ha="center", va="center",
             color="black", )#fontweight="bold")
    plt.text(x[i], nt_vals[i] + dy_vals[i]/2, f"{dy_vals[i]:,.0f}\n{dy_pct:.1f}%", ha="center", va="center",
             color="black",)# fontweight="bold")
plt.xticks(x, labels);plt.ylabel("Trip weight")
plt.tight_layout();plt.show()
#%% geometry related part
# MTA transit lines in Chicago
chitransitshape = pd.read_csv(r"C:\Users\Zhihua\Documents\MyResearch\NightTime\chicago\mdb-389-202505010221\shapes.txt")
shapes_grouped = chitransitshape.sort_values(['shape_id', 'shape_pt_sequence']).groupby('shape_id')
geometry = shapes_grouped.apply(lambda g: LineString(zip(g['shape_pt_lon'], g['shape_pt_lat'])), include_groups=False)
lines_gdf = gpd.GeoDataFrame(geometry, columns=['geometry'], crs="EPSG:4326")
lines_gdf = lines_gdf.to_crs(epsg=3435)
# match with census tract zone geometry
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
chizone = gpd.read_file(r"C:\Users\Zhihua\Documents\MyResearch\NightTime\chicago\tl_2020_17_tract\tl_2020_17_tract.shp")
chizone['centract'] = pd.to_numeric(chizone['GEOID'], errors='coerce').astype('Int64')
chint23 = dfchi23[(dfchi23.depart_hour>=20)|(dfchi23.depart_hour<6)]
chint23['o_tract_2020'] = pd.to_numeric(chint23['o_tract_2020'], errors='coerce').astype('Int64')
chint23['d_tract_2020'] = pd.to_numeric(chint23['d_tract_2020'], errors='coerce').astype('Int64')
chint23 = chint23.merge(chizone[['centract', 'geometry']], left_on='o_tract_2020', right_on='centract', how='left')
chint23 = chint23.rename(columns={'geometry': 'o_geom'}).drop(columns=['centract'])
chint23 = chint23.merge(chizone[['centract', 'geometry']], left_on='d_tract_2020', right_on='centract', how='left')
chint23 = chint23.rename(columns={'geometry': 'd_geom'}).drop(columns=['centract'])
chicitynt23 = chint23[~chint23['o_geom'].isna()]
chicitynt23 = chicitynt23[~chicitynt23['d_geom'].isna()]
# find centroid of census tracts
def to_point_safe(p):
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None  # keep as None or np.nan
    elif isinstance(p, Point):
        return p
    else:
        return Point(p)
chint23 = chint23.set_geometry('o_geom')
chint23 = chint23.set_geometry('d_geom')
chint23["centroid_o_x"] = chint23["o_geom"].centroid.apply(lambda p: p.x if p is not None else None)
chint23["centroid_o_y"] = chint23["o_geom"].centroid.apply(lambda p: p.y if p is not None else None)
chint23["centroid_d_x"] = chint23["d_geom"].centroid.apply(lambda p: p.x if p is not None else None)
chint23["centroid_d_y"] = chint23["d_geom"].centroid.apply(lambda p: p.y if p is not None else None)
chint23['o_area_ha'] = gpd.GeoSeries(chint23['o_geom']).to_crs(epsg=3435).area / 107639.1
chint23['d_area_ha'] = gpd.GeoSeries(chint23['d_geom']).to_crs(epsg=3435).area / 107639.1
chizone['area_1ha'] = gpd.GeoSeries(chizone['geometry']).to_crs(epsg=3435).area / 107639.1
chint23em = chint23[chint23['depart_hour']<6]
chint23ln = chint23[chint23['depart_hour']>=20]
#%% def gmm
# bins for the figure 
staH_bins = [0,1,2,3,4,5,20,21,22,23] #departure_hour
durationbins   = [0,10,20,30,45,60,90] #t.time_min
durationlabel = ["<10","10-20","20-30","30-45","45-60","60+"]
mode_bins = ['walk', 'micro', 'tran', 'taxi', 'auto'] #mainmode
purposebin = ['HBW','HBE','NHBW','HBS','HBR','HBO','NHBO']
# define cluster analysis function
def run_gmm_cluster_plot(df,gdf,lines_gdf, person21, valuemax, wo, wd, namefile):
    features=df[["centroid_o_x","centroid_o_y","centroid_d_x","centroid_d_y"]].dropna()
    df=df.loc[features.index].copy()
    Xo=StandardScaler().fit_transform(features[["centroid_o_x","centroid_o_y"]])*wo
    Xd=StandardScaler().fit_transform(features[["centroid_d_x","centroid_d_y"]])*wd
    X=np.hstack([Xo,Xd])    
    weights = df["tweight"].round().astype(int)
    X_rep = np.repeat(X, weights, axis=0)
    K=range(1,20);bic=[GaussianMixture(n_components=k,random_state=4).fit(X).bic(X) for k in K]
    best_K = K[np.argmin(bic)]
    gmm = GaussianMixture(n_components=best_K,random_state=4).fit(X)
    pred = gmm.predict(X);order=np.argsort(-np.bincount(pred))    
    sizes=np.bincount(gmm.predict(X));     order=np.argsort(-sizes)
    mapping=dict(zip(order,range(best_K)))
    labels=np.vectorize(mapping.get)(gmm.predict(X)) 
    min_size = len(df)/10
    sizes = np.bincount(pred)
    small_clusters = np.where(sizes < min_size)[0]
    for s in small_clusters:
        dists = np.linalg.norm(gmm.means_ - gmm.means_[s], axis=1)
        dists[s] = np.inf
        target = np.argmin(dists)
        pred[pred==s] = target
    df["cluster"] = np.vectorize(dict(zip(order,range(best_K))).get)(pred)
    clusters = sorted(df["cluster"].unique())
    cluster_map = {old: new for new, old in enumerate(clusters)}
    df["cluster"] = df["cluster"].map(cluster_map)
    min_size = len(df) / 10
    cluster_sizes = df["cluster"].value_counts()    
    small_after_merge = cluster_sizes[cluster_sizes < min_size].index    
    df = df[~df["cluster"].isin(small_after_merge)].copy()    
    clusters = sorted(df["cluster"].unique())
    cluster_map = {old: new for new, old in enumerate(clusters)}
    df["cluster"] = df["cluster"].map(cluster_map)
    df["cluster"] = pd.factorize(df["cluster"], sort=True)[0]
    best_K = df["cluster"].nunique()    #best_K = len(clusters)
    clusters = sorted(df["cluster"].unique())
    clusters_to_plot = clusters;    rowcount=1    #2 if best_K>5 else 1
    columncount=best_K  #-1#(best_K+1)//2 if best_K>5 else best_K #or use df["cluster"].max()
    trip_counts=df.groupby(["o_tract_2020","cluster"])['tweight'].sum().reset_index(name="trip_count")
    norm = mpl.colors.Normalize(vmin=0, vmax=valuemax)
    sm = mpl.cm.ScalarMappable(cmap="OrRd", norm=norm);    sm.set_array([])
    #OD maps
    fig,axs=plt.subplots(rowcount,columncount,figsize=(columncount*3,rowcount*3+2),squeeze =False);axs=axs.flatten()
    base_gdf = gdf[['geometry','area_1ha','centract']].copy()
    for i, c in enumerate(clusters_to_plot):
        plot_gdf = base_gdf.copy()
        cc=trip_counts[trip_counts["cluster"]==c][["o_tract_2020","trip_count"]].rename(columns={"trip_count":"cluster_trip_count_o"})
        plot_gdf = plot_gdf.merge(cc, left_on='centract', right_on='o_tract_2020', how='left').fillna({"cluster_trip_count_o": 0})
        plot_gdf["cluster_trip_count_o"] = plot_gdf["cluster_trip_count_o"].fillna(0)
        plot_gdf["density"]=plot_gdf.cluster_trip_count_o/plot_gdf.area_1ha/person21
        plot_gdf.plot(ax=axs[i],color = 'lightgrey',alpha = 0.5)
        plot_gdf.plot(column="density",cmap="OrRd",legend=False,ax=axs[i],vmin=0,vmax=valuemax,alpha = 0.8)
        lines_gdf.plot(ax=axs[i],linewidth=.03,alpha=.3)
        axs[i].set_title(f"O Cluster {c}");axs[i].axis("off")
    fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.02, pad=0.01, shrink = 0.6)    #plt.tight_layout();
    fig.savefig(f"O_clusters_density_{namefile}.png", dpi=300); plt.show()
    trip_counts=df.groupby(["d_tract_2020","cluster"])['tweight'].sum().reset_index(name="trip_count")    
    norm = mpl.colors.Normalize(vmin=0, vmax=valuemax)
    sm = mpl.cm.ScalarMappable(cmap="Blues", norm=norm);     sm.set_array([])
    fig,axs=plt.subplots(rowcount,columncount,figsize=(columncount*3,rowcount*3+2),squeeze =False);axs=axs.flatten()
    for i,c in enumerate(clusters_to_plot):
        plotd_gdf = base_gdf.copy()
        cc=trip_counts[trip_counts["cluster"]==c][["d_tract_2020","trip_count"]].rename(columns={"trip_count":"cluster_trip_count_d"})
        plotd_gdf = plotd_gdf.merge(cc, left_on='centract', right_on='d_tract_2020', how='left').fillna({"cluster_trip_count_d": 0})
        plotd_gdf["density"]=plotd_gdf.cluster_trip_count_d/plotd_gdf.area_1ha/person21
        plotd_gdf.plot(ax=axs[i],color = 'lightgrey',alpha = 0.5)
        plotd_gdf.plot(column="density",cmap="Blues",legend=False,ax=axs[i],vmin=0,vmax=valuemax,alpha = 0.8)
        lines_gdf.plot(ax=axs[i],linewidth=.03,alpha=.3)
        axs[i].set_title(f"D Cluster {c}");axs[i].axis("off")
    fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.02, pad=0.01, shrink = 0.6)    #plt.tight_layout();
    fig.savefig(f"D_clusters_density_{namefile}.png", dpi=300); plt.show()
    plt.rcParams.update({"font.size": 12,"axes.titlesize": 12,"axes.labelsize": 12,"xtick.labelsize": 12,"ytick.labelsize": 12,
    "legend.fontsize": 12})
    # stacked duparture time, purpose, and mode graph
    def stacked_pct(var,bins,title,fname):
        plot_data=[]
        for c in clusters_to_plot:
            d=df[df["cluster"]==c]    #['trip_weight'].sum().reset_index(name="trip_count")
            if isinstance(bins,list):counts=d.groupby(var).size().reindex(bins, fill_value=0)
            else:counts, _ = np.histogram(d[var].dropna(), bins=bins)      
            pct=counts/counts.sum()*100 if counts.sum()>0 else np.zeros(len(counts))
            plot_data.append(pct)
        plot_df=pd.DataFrame(plot_data,index=[f"C{c}" for c in clusters_to_plot])
        fig,ax=plt.subplots(figsize=(best_K*1,3))
        plot_df.plot(kind='bar',stacked=True,ax=ax,colormap="tab20", legend=False)
        ax.set_ylabel("Percentage (%)",fontsize = 11); plt.tight_layout();    #ax.set_title(title)
        fig.savefig(fname, dpi=300);plt.show()    
    stacked_pct("depart_hour",staH_bins,"Departure Time Distribution",f"stah_{namefile}.png")
    stacked_pct("trippur",purposebin,"Purpose Distribution",f"purpose_{namefile}.png")
    stacked_pct("mainmode",mode_bins,"Mode Distribution",f"mode_{namefile}.png")  
    # stacked duration graph
    def stacked_continuous_pct(var,bins,title,fname):
        plot_data=[]
        if isinstance(bins, (list, np.ndarray)):
            bin_len = len(bins) - 1  
        else:
            bin_len = bins  
        for c in clusters_to_plot:
            d = df[df["cluster"] == c][[var]].dropna()    
            if len(d) == 0:
                counts = np.zeros(bin_len)
            else:
                counts, _ = np.histogram(d[var].dropna(), bins=bins)
            pct = counts / counts.sum() * 100 if counts.sum() > 0 else np.zeros(bin_len)
            plot_data.append(pct)
        if isinstance(bins, (list,np.ndarray)):
            bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        else:
            bin_labels = [f"bin{i}" for i in range(len(plot_data[0]))]
        plot_df = pd.DataFrame(plot_data, index=[f"C{c}" for c in clusters_to_plot], columns=bin_labels)    
        fig,ax=plt.subplots(figsize=(best_K*1,3))
        plot_df.plot(kind='bar',stacked=True,ax=ax,colormap="Blues", legend=False)
        ax.set_ylabel("Percentage (%)",fontsize = 11);plt.tight_layout();#ax.set_title(title)
        fig.savefig(fname, dpi=300);plt.show()
    stacked_continuous_pct("duration_minutes",durationbins,"Duration Distribution",f"duration_{namefile}.png")   
    return df,best_K
#%%
chicity = gpd.read_file(r"C:\Users\Zhihua\Documents\MyResearch\NightTime\chicago\Census_20Tracts\Census_Tracts.shp")
chicity['area_1ha'] = gpd.GeoSeries(chicity['geometry']).to_crs(epsg=3435).area / 107639.1
chicity['centract'] = pd.to_numeric(chicity['CENSUS_T_1'], errors='coerce').astype('Int64')
os.chdir(r'C:\Users\Zhihua\Documents\MyResearch\NightTime\chicago')
lines_gdf = gpd.clip(lines_gdf, chicity)
emmuncluster,best_K=run_gmm_cluster_plot(chint23em, chicity, lines_gdf, 1, 4, 1, 1,"chiem")
ntmuncluster,best_K=run_gmm_cluster_plot(chint23ln, chicity, lines_gdf, 1, 4, 1, 1,"chiln")
