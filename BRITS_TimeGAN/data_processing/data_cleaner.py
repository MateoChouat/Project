# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:53:45 2025

@author: mc
"""
import pandas as pd
import geopandas as gpd
from knowledge import cm
import numpy as np


def merge_datasets(dfA, dfB, ID):
    geodf = gpd.merge(dfA, dfB, on=ID, how='inner')
    return geodf

def data_clean(gdf, dfA_cat_features, dfA_num_features, dfB_cat_features, dfB_num_features):
    cat_features = dfA_cat_features + dfB_cat_features
    num_features = dfA_num_features + dfB_num_features 
    gdf.set_crs(epsg=2154)
    gdf = gdf[~gdf['geometry'].isnull()]
    gdf[cat_features] = gdf[cat_features].astype(str).apply(lambda x: x.str.strip())
    if 'ANNEEPOSE' in gdf.columns:
        gdf['ANNEEPOSE'] = pd.to_datetime(gdf['ANNEEPOSE'], errors='coerce', format="%Y/%m/%d")
        gdf['ANNEEPOSE'] = gdf['ANNEEPOSE'].dt.strftime('%Y-%m-%d')

    if 'DATEREPA' in gdf.columns:
        gdf['DATEREPA'] = pd.to_datetime(gdf['DATEREPA'], errors='coerce', format="%Y/%m/%d")
        gdf['DATEREPA'] = gdf['DATEREPA'].dt.strftime('%Y-%m-%d')
    
    for i in gdf.columns:
        if i in cm:
            gdf[i] = gdf[i].fillna(cm[i])
    if 'DATEABANDON' not in gdf.columns:
        gdf['DATEABANDON'] = 0
    else:
        gdf['DATEABANDON'] = gdf['DATEABANDON'].fillna(0)
        
    gdf['ETAT'] = 'en service' if 'DATEABANDON' == 0 else 'hors service'
    cat_features.append('ETAT')

    gdf['LEN_WEIGHT'] = gdf['geometry'].apply(lambda x: np.log(x.length + 1))
    num_features.append('LEN_WEIGHT')
    print("✅ Dataset cleaned")
    return gdf, cat_features, num_features