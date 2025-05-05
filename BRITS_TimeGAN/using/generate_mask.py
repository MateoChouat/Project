import pandas as pd
import numpy as np

def generate_masks_deltas(geodataframe):
    """
    Cette fonction génère les masques et les deltas (et deltas inversés) pour chaque canalisation
    à partir du `geodataframe` qui contient les informations des observations.

    Args:
        geodataframe (pd.DataFrame) : DataFrame contenant les relevés des canalisations avec
                                      des timestamps et des informations de casse.

    Returns:
        masks (np.array) : Masques indiquant la présence ou l'absence de données.
        deltas (np.array) : Temps écoulé depuis la dernière observation.
        deltas_rev (np.array) : Temps écoulé depuis la prochaine observation.
    """
    
    # Trier par IDCANA et timestamp pour être sûr que les observations sont dans le bon ordre
    geodataframe = geodataframe.sort_values(by=['IDCANA', 'timestamp'])
    
    # Liste pour stocker les résultats
    masks = []
    deltas = []
    deltas_rev = []
    
    # Parcours de chaque canalisation unique (IDCANA)
    for idcana in geodataframe['IDCANA'].unique():
        # Extraire les données pour une canalisation spécifique
        canalisation_data = geodataframe[geodataframe['IDCANA'] == idcana]
        
        # Convertir les timestamps en datetime si ce n'est pas déjà le cas
        canalisation_data['timestamp'] = pd.to_datetime(canalisation_data['timestamp'])
        
        # Calculer les deltas (temps écoulé entre deux observations successives)
        delta_list = [0]  # Le premier delta est toujours 0 (aucune observation précédente)
        for i in range(1, len(canalisation_data)):
            delta = (canalisation_data['timestamp'].iloc[i] - canalisation_data['timestamp'].iloc[i-1]).total_seconds()
            delta_list.append(delta)
        
        # Calculer les deltas inversés (temps écoulé jusqu'à l'observation suivante)
        delta_rev_list = [0] * len(delta_list)  # Initialiser les deltas inversés à zéro
        for i in range(len(canalisation_data)-2, -1, -1):
            delta_rev = (canalisation_data['timestamp'].iloc[i+1] - canalisation_data['timestamp'].iloc[i]).total_seconds()
            delta_rev_list[i] = delta_rev
        
        # Créer les masques (1 si l'observation est présente, sinon 0)
        mask_list = [1 if not pd.isna(val) else 0 for val in canalisation_data['value']]  # Assumer qu'il y a une colonne 'value'
        
        # Ajouter les résultats de cette canalisation aux listes principales
        masks.extend(mask_list)
        deltas.extend(delta_list)
        deltas_rev.extend(delta_rev_list)

    # Convertir en arrays NumPy
    masks = np.array(masks)
    deltas = np.array(deltas)
    deltas_rev = np.array(deltas_rev)

    return masks, deltas, deltas_rev
