import os
configfile: "./workflow_config.json"
dataset = config["DATASETS"]
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_ANGLE"]

# DONE: currently this only works on one dataset, figure out a way to do it on 2 or more datasets
rule all:
    input:
        expand('../data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/{dataset}/',
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    run:
        # Because dataloader asks for the parent directory
        childdir = str(input)
        parentdir = os.path.dirname(os.path.dirname(childdir))
        shell("python train_model.py %s {config[network_name]}" % parentdir)


rule preprocess_training_images:
    input:
        expand("../data/raw/{dataset}_Training_Images/", dataset = config['DATASETS'])
    output:
        directory(expand("../data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/{dataset}/",
                         dataset = config['DATASETS'],
                         n_augmentation = config['N_AUGMENTATION'],
                         maxdegree = config['MAX_ROTATION_ANGLE']))
    run:
            shell("python ./utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree}")


rule preprocess_testing_images:
    input:
        expand("../data/raw/{dataset}_Testing_Images/", dataset = config['DATASETS'])
    output:
        directory(expand("../data/processed/testing/{dataset}/", dataset = config['DATASETS']))
    run:
        shell("python ./utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree}")


rule preprocess_annotations:
    input:
        "../data/raw/ODIR_Training_Annotations/ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    output:
        "../data/processed/annotations/ODIR_Annotations.csv"
    shell:
        "python preprocess_annotations.py {input} {output}"

# rule preprocess_annotations:
#     input:
#         # some input
#     output:
#         # some output
#     run:
#         # some command

# rule introspection:
#     input:
#         # some delicious input
#     output:
#         # some tasty output
#     run:
#         # some juicy command



