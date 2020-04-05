"""
Trigger training here
"""
import argparse
import os
import sys
import time

import torch
from torchvision import datasets, transforms
from utils.training import Decoder, Encoder, OdirVAETraining, VAEDataset
from utils.utils import setup


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = """Training VAE""")
    parser.add_argument('imfolder', type=str, default=None,
                        metavar='image_dir',
                        help="""The path to the directory which contains
                        the imgge folder. The images themselves must be
                        in one or more subdirectories of the imfolder""")
    parser.add_argument('path_prefix', type=str, default=None,
                        metavar='output_dir',
                        help="""Path to the directory where the torch and eventfiles
                        would be saved to""")
    parser.add_argument('networkname', type=str, default=None,
                        metavar='network_name',
                        help="""Path to the directory where the torch and eventfiles
                        would be saved to""")
    args, rest = parser.parse_known_args()

    FLAGS, logger = setup(running_script="./utils/training.py", args=rest, config="config.json")
    print("FLAGS= ", FLAGS)


    imfolder = os.path.abspath(args.imfolder)

    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    print("\ninput dir: ", imfolder,
          "\ndevice: ", device)

    os.makedirs(args.path_prefix, exist_ok=True)
    if args.networkname in os.listdir(args.path_prefix):
        input1 = input("\nNetwork already exists. Are you sure to proceed? ([y]/n) ")
        if not input1 in ['y', 'yes']:
            sys.exit()

    print("\nLoad Data as Tensors...")
    img_dataset = datasets.ImageFolder(imfolder, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    data = VAEDataset(img_dataset)

    encoder, decoder = Encoder(z=FLAGS.zdim), Decoder(z=FLAGS.zdim)

    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        path_prefix=args.path_prefix,
        network_name=args.networkname,
        device=device,
        optimizer_kwargs={"lr": FLAGS.learningrate},
        batch_size=FLAGS.batchsize,
        max_epochs=FLAGS.maxpochs,
        verbose=True,
    )

    print("\nSize of the dataset: {}\nShape of the single tensors: {}".format(len(data), data[0][0].shape))
    print("\nTo check if values are between 0 and 1:\n{}".format(data[0][0][0][50][30:180:10]))

    print("\nStart Training...")
    time_start = time.time()
    trained_encoder, _ = training.train()
    print('\nTraining with %i epochs done! Time elapsed: %.2f minutes' % (FLAGS.maxpochs, (time.time() - time_start)/60))
    # print(trained_encoder)

    # TODO: Also refactor path_prefix/networkname into args
    # Save network
    PATH = f'{args.path_prefix}/{args.networkname}/{args.networkname}.pth'
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(trained_encoder.state_dict(), PATH)


"""    
def prepare_datasets(logger, path_to_splits):
datasets = {'train': ''}
return datasets

FLAGS, logger = setup(running_script="train_ECG_vae.py",
                      config='config.json')

# input
split_data_path = FLAGS.input.strip().split(',')

datasets, eids = prepare_datasets(logger, split_data_path)

trained = train(logger, FLAGS, datasets['train'])

logger.info('Done.')
"""
