"""
Add plotting and introspection functions here
"""

import pandas as pd

import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    imfolder = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Testing_Images"
    csv_file = "/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv"
    csv_df = pd.read_csv(csv_file, sep='\t')

    list_not_suitable_jpgs_for_introspection_because_of_absence_in_annotations = []
    for jpg in tqdm(os.listdir(imfolder)):
        if len(csv_df.loc[csv_df['Fundus Image'] == jpg].index) == 0:
            list_not_suitable_jpgs_for_introspection_because_of_absence_in_annotations.append(jpg)
    list_not_suitable_jpgs_for_introspection_because_of_absence_in_annotations=list_not_suitable_jpgs_for_introspection_because_of_absence_in_annotations.sort()
    print(len(list_not_suitable_jpgs_for_introspection_because_of_absence_in_annotations))
    print(list_not_suitable_jpgs_for_introspection_because_of_absence_in_annotations)


