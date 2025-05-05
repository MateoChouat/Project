# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:35:22 2025

@author: mc
"""

import geopandas as gpd


def get_data(cana_path, defa_path):
    gdf_cana = gpd.read_file(cana_path)
    gdf_defa = gpd.read_file(defa_path)
    if gdf_cana.crs is None:
        gdf_cana.set_crs(epsg=2154, inplace=True)

    if gdf_defa.crs is None:
        gdf_defa.set_crs(epsg=2154, inplace=True)
    gdf_cana = gdf_cana.to_crs(epsg=2154)
    gdf_defa = gdf_defa.to_crs(epsg=2154)  
    
    print("✅ Datasets loaded")
    return gdf_cana, gdf_defa



