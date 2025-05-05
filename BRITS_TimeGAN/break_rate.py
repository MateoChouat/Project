# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:12:45 2025

@author: mc
"""


import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium import Choropleth
from branca.colormap import linear

def calculer_taux_casse(gdf, casse_col="casses_imputees", age_col="age_oper", len_col="SHAPE_LEN"):
    """
    Calcule le taux de casse pondéré par la longueur et l'âge de la canalisation.
    Taux = nb_casses / (longueur * âge)
    """
    gdf = gdf.copy()
    denominateur = gdf[len_col].replace(0, np.nan) * gdf[age_col].replace(0, np.nan)
    gdf["taux_casse"] = gdf[casse_col] / denominateur
    gdf["taux_casse"] = gdf["taux_casse"].fillna(0)
    return gdf


def generer_carte_folium(gdf, value_col="taux_casse", id_col="IDCANA"):
    """
    Affiche une carte Folium colorée selon le taux de casse.
    """
    gdf = gdf.copy()
    gdf = gdf.to_crs(epsg=4326)  # nécessaire pour folium

    # Colormap linéaire
    colormap = linear.OrRd_09.scale(gdf[value_col].min(), gdf[value_col].max())
    colormap.caption = "Taux de casse (par an)"

    # Carte centrée automatiquement
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
                   zoom_start=13, control_scale=True)

    # Ajout des canalisations avec couleur selon taux de casse
    for _, row in gdf.iterrows():
        color = colormap(row[value_col])
        folium.GeoJson(
            row.geometry,
            style_function=lambda _, c=color: {
                'color': c,
                'weight': 3,
                'opacity': 0.9
            },
            tooltip=folium.Tooltip(f"{id_col}: {row[id_col]}<br>Taux de casse: {row[value_col]:.3f}")
        ).add_to(m)

    colormap.add_to(m)
    return m