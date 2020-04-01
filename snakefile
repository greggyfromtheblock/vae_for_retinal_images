configfile: "./workflow_config.json"
dataset = config["DATASETS"]
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_DEGREE"]
split = config["SPLITS"]

# TODO: currently this only works on one dataset, figure out a way to do it on 2 datasets
rule all:
    input:
        parent=expand("../data/processed/{dataset}_Training_Images_n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/",
                      dataset = config['DATASETS'],
                      split= config['SPLITS'],
                      n_augmentation = config['N_AUGMENTATION'],
                      maxdegree = config['MAX_ROTATION_DEGREE']),
        child=expand("../data/processed/{dataset}_Training_Images_n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/images/",
                      dataset = config['DATASETS'],
                      split= config['SPLITS'],
                      n_augmentation = config['N_AUGMENTATION'],
                      maxdegree = config['MAX_ROTATION_DEGREE'])
    shell:
        "python train_model.py {input.parent} {config[network_name]}"


rule preprocess_images:
    input:
        expand("../data/raw/{dataset}_{split}_Images/", dataset = config['DATASETS'], split= config['SPLITS'])
    output:
        directory(expand("../data/processed/{dataset}_{split}_Images_n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/images/",
               dataset = config['DATASETS'],
               split= config['SPLITS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_DEGREE']))
    run:
        if split == 'Training' or split == 'training':
            shell("python ./utils/preprocessing.py {input} {output} -na {config[n_augmentation]} -mra {config[max_rotation_degree]}")
        else:
            shell("python ./utils/preprocessing.py {input} {output} -na 0 -mra 0")


rule preprocess_annotations:
    input:
        "../data/raw/ODIR_Training_Annotations/ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    output:
        "../data/processed/annotations/ODIR_Annotations.csv"
    shell:
        "python decode_diagnostics_keywords.py {input} {output}"
