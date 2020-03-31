configfile: "./workflow_config.json"
dataset = config["DATASETS"]
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_DEGREE"]

rule all:
    input:
        expand("../data/processed/{dataset}_{split}_Images_n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/",
               dataset = config['DATASETS'],
               split= config['SPLITS'],
               n_augmentation = config['SPLITS'],
               maxdegree = config['MAX_ROTATION_DEGREE'])
    shell:
        "python3 train_model.py {input} {config.network_name}"


rule preprocess_images:
    input:
        expand("../data/raw/{dataset}_{split}_Images/", dataset = config['DATASETS'], split= config['SPLITS'])
    output:
        expand("../data/processed/{dataset}_{split}_Images_n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/",
               dataset = config['DATASETS'],
               split= config['SPLITS'],
               n_augmentation = config['SPLITS'],
               maxdegree = config['MAX_ROTATION_DEGREE'])
    run:
        if wildcards.split == 'Training' or wildcards.split == 'training':
            shell("python3 ./utils/preprocessing.py {input} {output} {config[n_augmentation]} {config[max_rotation_degree]}")
        else:
            shell("python3 ./utils/preprocessing.py {input} {output} 0 0")


rule preprocess_annotations:
    input:
        "../data/raw/annotations/ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    output:
        "../data/processed/annotations/ODIR_Annotations.csv"
    shell:
        "python3 decode_diagnostics_keywords.py {input} {output}"
