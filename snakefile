# TODO: Move variables to the config file
DATASETS = ["ODIR", "PALM"]
SPLITS = ["Training", "Testing"]
RESIZE_DIMENSIONS = [0,0]


rule train:
    input:
        "../data/processed/ODIR_Training_Images/"
    shell:
        "python3 -u train_model.py {input}"


rule preprocess_images:
    input:
        expand("../data/raw/{dataset}_{split}_Images/", dataset = DATASETS, split= SPLITS)
    output:
        expand("../data/processed/{dataset}_{split}_Images/", dataset = DATASETS, split= SPLITS)
    shell:
        "python3 -u ./utils/preprocessing.py --resize resize_dimensions[0] resize_dimensions[1]  {input} {output}"

rule preprocess_annotations:
    input:
        "../data/raw/ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    output:
        "../data/processed/ODIR_Annotations.csv"
    conda:
        "../envs/greg.yml"
    shell:
        "python3 -u decode_diagnostics_keywords.py {input} {output}"
