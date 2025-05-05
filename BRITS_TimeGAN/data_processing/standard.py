# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:35:37 2025

@author: mc
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd

def data_standardized(gdf, num_features, cat_features):
    scaler = StandardScaler
    gdf[num_features] = scaler.fit_transform(gdf[num_features])
    gdf = pd.get_dummies(gdf, columns=[cat_features], dummy_na = False)
    return gdf, scaler


