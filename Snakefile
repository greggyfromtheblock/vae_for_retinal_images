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
resize1 = config['RESIZE'][0]
resize2 = config['RESIZE'][1]
grayscale = config['GRAYSCALE']

rule all:
    input:
        "mybody.done"
    run:
        path = str(path_prefix) + str(networkname)
        shell("tensorboard --logdir %s --port {port}" % path)

rule introspection:
    input:
        dummyfile = "mytask.done",
        annotations = "/home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv",
        imdir = expand("/home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/testing/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
                     dataset = config['DATASETS'],
                     n_augmentation = config['N_AUGMENTATION'],
                     maxdegree = config['MAX_ROTATION_ANGLE'],
                     resize1 = config['RESIZE'][0],
                     resize2 = config['RESIZE'][1],
                     grayscale = config['GRAYSCALE'])
    output:
        touch('mybody.done')
    run:
        shell('python utils/introspection.py -i {input.imdir} -csv {input.annotations} -pp {path_prefix} -nn {networkname} -md {maxdegree}')


rule training:
    input:
        expand("/home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
                         dataset = config['DATASETS'],
                         n_augmentation = config['N_AUGMENTATION'],
                         maxdegree = config['MAX_ROTATION_ANGLE'],
                         resize1 = config['RESIZE'][0],
                         resize2 = config['RESIZE'][1],
                         grayscale = config['GRAYSCALE'])
    output:
        touch("mytask.done")
    run:
        shell("python train_model.py -i {input} -pp {path_prefix} -nn {networkname}" )


rule preprocess_training_images:
    input:
        expand("/home/henrik/PycharmProjects/vae_for_retinal_images/data/odir/{dataset}_Training_Images/", dataset = config['DATASETS'])
    output:
        expand("/home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
                         dataset = config['DATASETS'],
                         n_augmentation = config['N_AUGMENTATION'],
                         maxdegree = config['MAX_ROTATION_ANGLE'],
                         resize1 = config['RESIZE'][0],
                         resize2 = config['RESIZE'][1],
                         grayscale = config['GRAYSCALE'])
    run:
         shell("python utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {resize1} {resize2} -gr {grayscale}")


rule preprocess_testing_images:
    input:
        expand("/home/henrik/PycharmProjects/vae_for_retinal_images/data/odir/{dataset}_Test_Images/", dataset = config['DATASETS'])
    output:
       expand("/home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/testing/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
                         dataset = config['DATASETS'],
                         n_augmentation = config['N_AUGMENTATION'],
                         maxdegree = config['MAX_ROTATION_ANGLE'],
                         resize1 = config['RESIZE'][0],
                         resize2 = config['RESIZE'][1],
                         grayscale = config['GRAYSCALE'])
    run:
        shell("python utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {resize1} {resize2} -gr 0")


rule preprocess_annotations:
    input:
        "/home/henrik/PycharmProjects/vae_for_retinal_images/data/odir/ODIR_Annotations/ODIR-5K_Training_Annotations.xlsx"
    output:
        "/home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv"
    run:
        shell("python utils/preprocess_annotations.py {input} {output}")