import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    dir_to_save_histograms = "/data/analysis/ag-reils/ag-reils-shared-students/henrik2/vae_for_retinal_images/data/histograms"
    os.makedirs(dir_to_save_histograms, exist_ok=True)

    inspect_dir = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Testing_Introspection_Images"
    train_dir = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Training_Images"
    csv_file = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv"  # "/home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/annotations/annotations.csv"

    csv_df = pd.read_csv(csv_file, sep='\t')

    diagnoses = {
        "N": "normal fundus",
        "D": "proliferative retinopathy",
        "G": "glaucoma",
        "C": "cataract",
        "A": "age related macular degeneration",
        "H": "hypertensive retinopathy",
        "M": "myopia",
        "O": "others"
    }

    print("Count frequencies of all diagnoses + the Patient Sex of all Images (contained in the Annotations)")
    diagnoses_list = list(diagnoses.keys())
    count_arr = np.zeros(len(diagnoses_list)+2)
    for j, feature in enumerate(diagnoses_list):
        number = csv_df[feature].sum()
        count_arr[j] = number
        print("Number of Patients with diagnosis {}: {}".format(diagnoses[feature], number))

    female, male = 0, 0
    nr_ag_groups = 5
    age_groups = [0 for n in range(0, 100, nr_ag_groups)]
    for index, row in csv_df.iterrows():
        if row["Patient Sex"] == "Female":
            female += 1
        else:
            male += 1

        age = row["Patient Age"]
        age_groups[age//nr_ag_groups] += 1

    count_arr[-2] = female
    count_arr[-1] = male

    print("\nNumber of Females: {}\nNumber of Males: {}\n".format(male//2, female//2))
    for age_group in range(0, 100, nr_ag_groups):
        print("Number of Patients of Age-Group {} - {}: {}".format(age_group, age_group + 4, age_groups[age_group//nr_ag_groups]))

    diagnoses_list2 = list(diagnoses.values())
    diagnoses_list2.append("Female")
    diagnoses_list2.append("Male")

    colors = ['navy', 'firebrick', 'darkorange', 'indigo',  'cornflowerblue', 'darkgreen',
              'red', 'sienna','turquoise' , 'limegreen']

    plt.xticks(rotation='vertical')
    plt.bar(diagnoses_list2, count_arr, color=colors)
    plt.ylabel("Häufigkeit", fontsize=14, fontweight='bold')
    plt.title("Frequencies of all diagnoses + the Patient Sex of all Images (contained in the Annotations)")
    plt.savefig(dir_to_save_histograms+"frequencies_all_Images.png")
    plt.show()
    plt.close()


    print("\n\nCount frequencies of all diagnoses + the Patient Sex of the Testing Introspection Images")
    count_arr = np.zeros(len(diagnoses_list2))
    marker = None
    for i, jpg in enumerate(os.listdir(train_dir)):
        if jpg == '.snakemake_timestamp':
            marker = True
            continue
        try:
            row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]
            for j, feature in enumerate(diagnoses_list):
                if not marker:
                    count_arr[j] += csv_df.iloc[row_number].at[feature]
                else:
                    count_arr[j] += csv_df.iloc[row_number].at[feature]
            if csv_df.iloc[row_number].at["Patient Sex"] == "Female":
                count_arr[-2] += 1
            else:
                count_arr[-1] += 1
        except IndexError:
            pass

    plt.xticks(rotation='vertical')
    plt.bar(diagnoses_list2, count_arr, color=colors)
    plt.ylabel("Häufigkeit", fontsize=14, fontweight='bold')
    plt.title("Frequencies of all diagnoses + the Patient Sex of all Images (contained in the Annotations)")
    plt.savefig(dir_to_save_histograms+"frequencies_training_images.png")
    plt.show()
    plt.close()


    print("\n\nCount frequencies of all diagnoses + the Patient Sex of the Training Images")
    count_arr = np.zeros(len(diagnoses_list2))
    marker = None
    for i, jpg in enumerate(os.listdir(inspect_dir)):
        if jpg == '.snakemake_timestamp':
            marker = True
            continue
        try:
            row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]
            for j, feature in enumerate(diagnoses_list):
                if not marker:
                    count_arr[j] += csv_df.iloc[row_number].at[feature]
                else:
                    count_arr[j] += csv_df.iloc[row_number].at[feature]
            if csv_df.iloc[row_number].at["Patient Sex"] == "Female":
                count_arr[-2] += 1
            else:
                count_arr[-1] += 1
        except IndexError:
            pass

    plt.xticks(rotation='vertical')
    plt.bar(diagnoses_list2, count_arr, color=colors)
    plt.ylabel("Häufigkeit", fontsize=14, fontweight='bold')
    plt.title("Frequencies of all diagnoses + the Patient Sex of all Testing Introspection Images")
    plt.savefig(dir_to_save_histograms+"frequencies_introspection_images.png")
    plt.show()
    plt.close()
