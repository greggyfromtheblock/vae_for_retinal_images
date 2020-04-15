"""
to run this, modify the config file as you wish and run `snakemake --cores <cores>`
note that the raw annotations file need to be renamed to '../data/processed/annotations/ODIR_Annotations.csv'
"""

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
user = config['YOURNAME']

rule all:
    input:
        expand("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/mybody", user = config['YOURNAME'])
    run:
        path = "/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images" + str(path_prefix)[2:] + str(networkname)
        print("path to the eventfiles: %s" % path) 
        # shell("tensorboard --logdir %s --port {port}" % path)

rule introspection:
    input:
        dummyfile = "/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/mytask.done",
        annotations = expand("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv", user=config["YOURNAME"]),
        imdir = expand("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/data/processed/testing/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
             user = config["YOURNAME"],
             dataset = config['DATASETS'],
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             grayscale = config['GRAYSCALE'])
    output:
        touch('/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/mybody')
    run:
        shell('python utils/introspection.py -i {input.imdir} -csv {input.annotations} -pp {path_prefix} -nn {networkname} -md {maxdegree}')


rule training:
    input:
        expand("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
                 user = config["YOURNAME"],
                 dataset = config['DATASETS'],
                 n_augmentation = config['N_AUGMENTATION'],
                 maxdegree = config['MAX_ROTATION_ANGLE'],
                 resize1 = config['RESIZE'][0],
                 resize2 = config['RESIZE'][1],
                 grayscale = config['GRAYSCALE'])
    output:
        touch("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/mytask.done")
    run:
        shell("python train_model.py -i {input} -pp {path_prefix} -nn {networkname}" )


rule preprocess_training_images:
    input:
        expand("/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/{dataset}_Training_Images/", dataset = config['DATASETS'])
    output:
        directory(expand("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
                 user = config["YOURNAME"],
                 dataset = config['DATASETS'],
                 n_augmentation = config['N_AUGMENTATION'],
                 maxdegree = config['MAX_ROTATION_ANGLE'],
                 resize1 = config['RESIZE'][0],
                 resize2 = config['RESIZE'][1],
                 grayscale = config['GRAYSCALE']))
    run:
         shell("python utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {resize1} {resize2} -gr {grayscale}")


rule preprocess_testing_images:
    input:
        expand("/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/{dataset}_Testing_Introspection_Images", dataset = config['DATASETS'])
    output:
       directory(expand("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/data/processed/testing/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_grayscale_{grayscale}/{dataset}/",
             user = config["YOURNAME"],
             dataset = config['DATASETS'],
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             grayscale = config['GRAYSCALE']))
    run:
        shell("python utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {resize1} {resize2} -gr 0")


rule preprocess_annotations:
    input:
        "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Training_Annotations/ODIR-5K_Training_Annotations.xlsx"
    output:
        expand("/data/analysis/ag-reils/ag-reils-shared-students/{user}/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv",
        user=config["YOURNAME"])
    run:
        shell("python utils/preprocess_annotations.py {input} {output}")

