import os
configfile: "./workflow_config.json"
dataset = config["DATASETS"]
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_ANGLE"]

# TODO: test if this flag works on training
# TODO: still have to make argparse for input though
# FLAGS, _ = setup(running_script="./utils/training.py", config="config.json")


rule all:
    input:
        expand('../data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}/{dataset}/',
               dataset = config['DATASETS'],
               n_augmentation = config['N_AUGMENTATION'],
               maxdegree = config['MAX_ROTATION_ANGLE'])
    # output:
    #     "..%s/%s" % (FLAGS.path_prefix, FLAGS.networkname)
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



