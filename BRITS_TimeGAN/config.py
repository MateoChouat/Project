# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:18:14 2025

@author: mc
"""
import torch
from main import main

DATA_CANA_PATH = "C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/cana.shp"
DATA_DEFA_PATH = "C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/def.shp"

OUTPUT_REPAIRED_SERIES = "C:/Users/mc/Desktop/'BRITS_TimeGAN'/outputs/repaired_series/"
OUTPUT_GENERATED_HISTORY = "C:/Users/mc/Desktop/'BRITS_TimeGAN'/outputs/generated_history/"
OUTPUT_SHAPEFILES_FINALS = "C:/Users/mc/Desktop/'BRITS_TimeGAN'/outputs/shapefiles_finals/"

Year_Start = 1890
Year_End = 1989
MISSING_YEARS = list(range(Year_Start, Year_End + 1))

Year_MDWindow_Start = 1996
Year_MDWindow_End = 1999
MISSING_YEARS_MDWindow = list(range(Year_MDWindow_Start, Year_MDWindow_End + 1))

ID_Column = 'IDCANA'

device = torch.device('cuda' if torch.is_available('cuda') else 'cpu')


cana_num_features=[]
cana_cat_features=[]
defa_num_features=[]
defa_cat_features=[]
target_feature=''

if __name__ == '__main__':
    main()


