# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:44:33 2025

@author: mc
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import os


def saw_cana(gdf_cana, path_save, longueur_voulue): 
    def tronconner_ligne(geom, longueur_cible):
        if geom.length <= longueur_cible:
            return [geom]
        points = [geom.interpolate(d) for d in np.arange(0, geom.length, longueur_cible)]
        points.append(geom.interpolate(geom.length))
        
        segments = []
        for i in range(len(points) - 1):
            segments.append(LineString([points[i], points[i+1]]))
        return segments
    
    nouvelle_geoms = []
    nouveaux_attributs = []
    
    for idx, row in gdf_cana.iterrows():
        segments = tronconner_ligne(row.geometry, longueur_voulue)
        for i, seg in enumerate(segments):
            new_row = row.drop('geometry').copy()
            # Création d'un nouvel ID basé sur l'ancien
            new_row['IDCANA_INDUIT'] = f"{row['IDCANA']}_{i}"  # Créer un ID unique
            nouvelle_geoms.append(seg)
            nouveaux_attributs.append(new_row)
    
    cana_sawn = gpd.GeoDataFrame(nouveaux_attributs, geometry=nouvelle_geoms, crs=gdf_cana.crs)
    
    if path_save:
        os.makedirs(path_save, exist_ok=True)  # Crée le dossier si il n'existe pas
        output_path = os.path.join(path_save, 'cana_sawn.shp')
        cana_sawn.to_file(output_path)
        print(f"✅ Canalisations tronçonnées sauvegardées dans : {output_path}")
    
    return cana_sawn


def group_cana_defa(gdf_cana_sawn, gdf_defa, path_save):
    defa_sawn = gpd.sjoin(gdf_defa, gdf_cana_sawn[['IDCANA_INDUIT', 'geometry']], how='left', predicate='intersects')
    
    # Check si il y a des défauts non associés
    n_missing = defa_sawn['IDCANA'].isna().sum()
    if n_missing > 0:
        print(f"⚠️ Attention : {n_missing} défaut(s) non associés à une canalisation tronçonnée.")

    if path_save:
        os.makedirs(path_save, exist_ok=True)  # Crée le dossier si besoin
        output_path = os.path.join(path_save, 'defa_sawn.shp')
        defa_sawn.to_file(output_path)
        print(f"✅ Défauts associés sauvegardés dans : {output_path}")
    
    return defa_sawn
