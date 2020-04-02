import os

configfile: "./workflow_config.json"
dataset = config["DATASETS"]
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_DEGREE"]
split = config["SPLITS"]


rule introspection:
    input:
        "../data/processed/annotations/ODIR_Annotations.csv"
    output:
        # outputs a plot or something
    shell:
        "introspection.py"

# TODO: currently this only works on one dataset, figure out a way to do it on 2 datasets
# TODO: have a plot as the input of this
# TODO: rule training: fake a file
# TODO: introspection poops out the plot
rule all:
    input:
        expand("../data/processed/{dataset}_Training_Images_n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/images/",
                      dataset = config['DATASETS'],
                      n_augmentation = config['N_AUGMENTATION'],
                      maxdegree = config['MAX_ROTATION_DEGREE'])
    run:
        shell("python train_model.py {parent_dir} {config[network_name]}")


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
        "python preprocess_annotations.py {input} {output}"
