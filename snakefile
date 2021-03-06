"""
to run this, modify the config file as you wish and run `snakemake --cores <cores>`
note that the raw annotations file need to be renamed to '../data/processed/annotations/ODIR_Annotations.csv'
networkname here is not the network name that will be written, its just an identifier
"""

import os
from utils.utils import setup

configfile: "./workflow_config.json"
dataset = config["DATASETS"]
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_ANGLE"]
path_prefix = config['PATH_PREFIX']
networkname = config['NETWORKNAME']
port = config['PORT']
resize_dimensions = config['RESIZE_DIMENSIONS']

rule all:
    input:
        expand('../dummyfiles/dummyoutput_introspection{networkname}{dataset}{n_augmentation}{maxdegree}.txt',
               networkname = config['NETWORKNAME'],
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    run:
        FLAGS, _ = setup(running_script="./utils/introspection.py", config="config.json")
        networkpath = str(FLAGS.path_prefix) + '/' + str(FLAGS.networkname)
        shell("tensorboard --logdir %s --port {port}" % networkpath)

rule introspection:
    input:
        dummyfile = expand('../dummyfiles/dummyoutput_training{networkname}{dataset}{n_augmentation}{maxdegree}.txt',
                           dataset = config['DATASETS'],
                           n_augmentation = config['N_AUGMENTATION'],
                           maxdegree = config['MAX_ROTATION_ANGLE'],
                           networkname = config['NETWORKNAME']),
        annotations = "../data/processed/annotations/ODIR_Annotations.csv",
        imdir = expand("../data/processed/testing/{dataset}/images/", dataset = config['DATASETS'])

    output:
        touch(expand('../dummyfiles/dummyoutput_introspection{networkname}{dataset}{n_augmentation}{maxdegree}.txt',
               networkname = config['NETWORKNAME'],
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE']))
    run:
        childdir = str(input.imdir)
        parentdir = os.path.dirname(os.path.dirname(childdir))
        shell('python utils/introspection.py %s {input.annotations}' % parentdir)

rule training:
    input:
        expand("../data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/{dataset}/",
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    output:
        touch(expand('../dummyfiles/dummyoutput_training{networkname}{dataset}{n_augmentation}{maxdegree}.txt',
                     dataset = config['DATASETS'],
                     n_augmentation = config['N_AUGMENTATION'],
                     maxdegree = config['MAX_ROTATION_ANGLE'],
                     networkname = config['NETWORKNAME']))
    run:
        # Because dataloader asks for the parent directory
        childdir = str(input)
        parentdir = os.path.dirname(os.path.dirname(childdir))
        shell("python train_model.py %s" % parentdir)

rule preprocess_training_images:
    input:
        expand("../data/raw/{dataset}_Training_Images/", dataset = config['DATASETS'])
    output:
        directory(expand("../data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/{dataset}/",
                         dataset = config['DATASETS'],
                         n_augmentation = config['N_AUGMENTATION'],
                         maxdegree = config['MAX_ROTATION_ANGLE']))
    run:
        shell("python ./utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {config[RESIZE_DIMENSIONS][0]} {config[RESIZE_DIMENSIONS][1]}")


rule preprocess_testing_images:
    input:
        expand("../data/raw/{dataset}_Testing_Images/", dataset = config['DATASETS'])
    output:
        directory(expand("../data/processed/testing/{dataset}/images/", dataset = config['DATASETS']))
    run:
        shell("python ./utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {config[RESIZE_DIMENSIONS][0]} {config[RESIZE_DIMENSIONS][1]}")


rule preprocess_annotations:
    input:
        "../data/raw/ODIR_Training_Annotations/ODIR-5K_Training_Annotations.xlsx"
    output:
        "../data/processed/annotations/ODIR_Annotations.csv"
    run:
        shell("python utils/preprocess_annotations.py {input} {output}")

