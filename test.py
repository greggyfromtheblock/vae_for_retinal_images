"""
Add plotting and introspection functions here
"""

import pandas as pd
from skimage import io
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    test_introspection_folder = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Testing_Introspection_Images"
    test_folder = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Testing_Images"
    train_folder = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Training_Images"
    csv_file = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv"
    csv_df = pd.read_csv(csv_file, sep='\t')

    # not_suitable_jpgs_for_introspection_because_of_absence_in_annotations = []
    os.makedirs(test_introspection_folder, exist_ok=True)
    for i, jpg in tqdm(enumerate(os.listdir(train_folder))):
        if len(csv_df.loc[csv_df['Fundus Image'] == jpg].index) == 0 and 1000 < i < 1250 or 2200 < i < 2450 or 2900 < i < 3300\
                or 4000 < i < 4100:
            os.system(f'mv {train_folder}/{jpg} {test_introspection_folder}')

    print("size of train folder: %i" % len(os.listdir(train_folder)))
    print("size of test folder: %i" % len(os.listdir(test_folder)))
    print("size of test introspection folder: %i" % len(os.listdir(test_introspection_folder)))

    """
    if len(csv_df.loc[csv_df['Fundus Image'] == jpg].index) == 0:
        not_suitable_jpgs_for_introspection_because_of_absence_in_annotations.append(jpg)
        os.system(f'mv {jpg} {train_folder}')
    not_suitable_jpgs_for_introspection_because_of_absence_in_annotations.sort()
    print(len(not_suitable_jpgs_for_introspection_because_of_absence_in_annotations), "\n\n")
    print(not_suitable_jpgs_for_introspection_because_of_absence_in_annotations)
    """
