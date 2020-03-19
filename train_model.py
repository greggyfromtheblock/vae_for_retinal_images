"""
Trigger training here
"""

from utils.utils import setup


def prepare_datasets(logger, path_to_splits):
    datasets = {'train': '')
    return datasets


if name == '__main__':
    FLAGS, logger = setup(running_script="train_ECG_vae.py",
                          config='config.json')

    # input
    split_data_path = FLAGS.input.strip().split(',')

    datasets, eids = prepare_datasets(logger, split_data_path)

    trained = train(logger, FLAGS, datasets['train'])

    logger.info('Done.')