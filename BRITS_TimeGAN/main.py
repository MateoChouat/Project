# -*- coding: utf-8 -*-
"""
Created on Mon May  5 08:38:46 2025

@author: mc
"""

from data_processing import data_loader, data_cleaner, saw
from training import train_brits
from using import brits_using, generate_mask
from break_prediction import break_predict


def main(cana_path, defa_path, path_sawn_save, longueur_voulue, ID_Column, cana_cat_features, defa_cat_features, cana_num_features, defa_num_features):
    gdf_cana, gdf_defa = data_loader.get_data(cana_path, defa_path)
    
    #Normalisation de la longueur
    gdf_cana = saw.saw_cana(gdf_cana, path_sawn_save, longueur_voulue)
    gdf_defa = saw.group_cana_defa(gdf_cana, gdf_defa, path_sawn_save)
    
    
    #Utilisation des connaissances métiers pour imputation des MD
    gdf = data_cleaner.merge_datasets(gdf_cana, gdf_defa, ID_Column)
    gdf, gdf_cat_features, gdf_num_features = data_cleaner.data_clean(gdf, cana_cat_features, cana_num_features, defa_cat_features, defa_num_features)
    
    
    
    
    
