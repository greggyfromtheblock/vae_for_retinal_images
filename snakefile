"""
to run this, modify the config file as you wish and run `snakemake --cores <cores>`
note that the raw annotations file need to be renamed to '../data/processed/annotations/ODIR_Annotations.csv'
"""

import os

configfile: "./workflow_config.json"
dataset = config["DATASETS"]
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_ANGLE"]
path_prefix = config['PATH_PREFIX']
networkname = config['NETWORKNAME']
port = config['PORT']

rule all:
    input:
        expand('../dummyfiles/dummy{port}{networkname}{dataset}{n_augmentation}{maxdegree}.txt',
               port = config['PORT'],
               networkname = config['NETWORKNAME'],
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    run:
        path = str(path_prefix) + str(networkname)
        shell("tensorboard --logdir %s --port {port}" % path)

rule introspection:
    input:
        dummyfile = expand('../dummyfiles/dummy{dataset}{n_augmentation}{maxdegree}.txt',
                           dataset = config['DATASETS'],
                           n_augmentation = config['N_AUGMENTATION'],
                           maxdegree = config['MAX_ROTATION_ANGLE']),
        annotations = "../data/processed/annotations/ODIR_Annotations.csv",
        imdir = expand("../data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/ODIR/",
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    output:
        expand('../dummyfiles/dummy{port}{networkname}{dataset}{n_augmentation}{maxdegree}.txt',
               port = config['PORT'],
               networkname = config['NETWORKNAME'],
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    run:
        shell('python utils/introspection.py {input.imdir} {path_prefix} {input.annotations} {networkname}')
        touch(expand('../dummyfiles/dummy{port}{networkname}{dataset}.txt',
                     port = config['PORT'],
                     networkname = config['NETWORKNAME'],
                     dataset = config['DATASETS'],
                     n_augmentation = config['N_AUGMENTATION'],
                     maxdegree = config['MAX_ROTATION_ANGLE']))

rule training:
    input:
        expand("../data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/{dataset}/",
                         dataset = config['DATASETS'],
                         n_augmentation = config['N_AUGMENTATION'],
                         maxdegree = config['MAX_ROTATION_ANGLE'])
    output:
        expand('../dummyfiles/dummy{dataset}{n_augmentation}{maxdegree}.txt',
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    run:
        # Because dataloader asks for the parent directory
        childdir = str(input)
        parentdir = os.path.dirname(os.path.dirname(childdir))
        shell("python train_model.py %s {path_prefix} {networkname}" % parentdir)
        touch(expand('../dummyfiles/dummy{dataset}{n_augmentation}{maxdegree}.txt',
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE']))

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
        "../data/raw/ODIR_Training_Annotations/ODIR-5K_Training_Annotations.xlsx"
    output:
        "../data/processed/annotations/ODIR_Annotations.csv"
    run:
        shell("python utils/preprocess_annotations.py {input} {output}")




