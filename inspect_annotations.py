import pandas as pd

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
}

diagnoses_list = list(diagnoses.keys())
# diagnoses_list.extend(["Patient Sex"])
count_dict = {}
for j, feature in enumerate(diagnoses_list):
    number = csv_df[feature].sum()
    print("Number of Patients with diagnosis {}: {}".format(diagnoses[feature], number))
    # count_dict[diagnoses[feature]] = number

female, male = 0, 0
age_groups = [0 for n in range(0, 100, 5)]
for index, row in csv_df.iterrows():
    if row["Patient Sex"] == "Female":
        female += 1
    else:
        male += 1

    age = row["Patient Age"]
    if age < 5:
        age_groups[0] += 1
    elif age < 10:
        age_groups[1] += 1
    elif age < 15:
        age_groups[2] += 1
    elif age < 20:
        age_groups[3] += 1
    elif age < 25:
        age_groups[4] += 1
    elif age < 30:
        age_groups[5] += 1
    elif age < 35:
        age_groups[6] += 1
    elif age < 40:
        age_groups[7] += 1
    elif age < 45:
        age_groups[8] += 1
    elif age < 50:
        age_groups[9] += 1
    elif age < 55:
        age_groups[10] += 1
    elif age < 60:
        age_groups[11] += 1
    elif age < 65:
        age_groups[12] += 1
    elif age < 70:
        age_groups[13] += 1
    elif age < 75:
        age_groups[14] += 1
    elif age < 80:
        age_groups[15] += 1
    elif age < 85:
        age_groups[16] += 1
    elif age < 90:
        age_groups[17] += 1
    elif age < 95:
        age_groups[18] += 1
    else:
        age_groups[19] += 1

print("\n Number of Females: {}\nNumber of Males: {}\n".format(male//2, female//2))
for age_group in range(0, 100, 5):
    print("Number of Patients of Age-Group {} - {}: {}".format(age_group, age_group + 4, age_groups[age_group//5]))
