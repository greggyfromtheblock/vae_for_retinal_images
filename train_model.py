"""
Trigger training here
"""
import argparse
import os
import sys
from torchvision import datasets, transforms
import numpy as np
import torch
from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset
import time

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Training VAE""")
    parser.add_argument('imfolder', type=str, default=None, metavar='image_dir',
                        help="""The path to the directory which contains the imgge folder. The images themselves must be
                        in one or more subdirectories of the imfolder""")
    parser.add_argument('--path_prefix', type=str, default=".", metavar='path_prefix',
                        help="""The path to the directory which should contain the data for tensorboard.""")
    parser.add_argument('network_name', type=str, default=None, metavar='network_name',
                        help="""The name of the network. Use different names for different models!""")
    parser.add_argument('--gpu_number', type=int, default=3, metavar='gpu_number',
                        help="""The gpu you want to use.     Number must be between 0 and 7""")
    parser.add_argument('--n_epochs', type=int, default=100, metavar='n_epochs',
                        help="""Maximal number of epochs.""")
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help="""Size of a Batch.""")
    parser.add_argument('--learning_rate', type=float, default=5e-5,
        metavar='learning_rate',
                    help="""The learning_rate.""")
    args = parser.parse_args()

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

    imfolder = add_slash(args.imfolder)
    network_name = args.network_name
    path_prefix = args.path_prefix  # optional argument. If default: the path is the current one.
    gpu_number = int(args.gpu_number) if int(args.gpu_number) in range(8) else 3
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    if network_name in os.listdir(path_prefix):
        input1 = input("Network already exists. Are you sure to proceed? ([y]/n)\n")
        if not input1 in ['y', 'yes']:
            sys.exit()

    print("Load Data as Tensors...")
    transform_data = transforms.Compose([transforms.ToTensor(), normalize])
    img_dataset = datasets.ImageFolder(
        imfolder, transform=transform_data
    )

    data = VAEDataset(img_dataset)
    encoder, decoder = Encoder(), Decoder()
    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        path_prefix=path_prefix,
        network_name=network_name,
        optimizer_kwargs={"lr": lr},
        device="cuda:%i" % gpu_number if torch.cuda.is_available() else "cpu",
        batch_size=batch_size,
        max_epochs=n_epochs,
        verbose=True
    )
    print(len(data), data[0][0].shape)
    print("To check if values are between 0 and 1:\n", data[0][0][0][50][30:180:10])

    print("Start Training...")
    time_start = time.time()
    trained = training.train()
    print('\nTraining with %i done! Time elapsed: %.2f minutes' % (n_epochs, (time.time() - time_start)/60))
    trained_encoder, _ = training.train()
    # print(trained_encoder)

    # Save network
    PATH = f'{path_prefix}/{network_name}/{network_name}.pth'
    torch.save(trained_encoder.state_dict(), PATH)


    print(type(features))
    if type(features) == torch.Tensor:
        torch.save(features, f"{path_prefix}/{network_name}/features.pt")
        torch.save(mean, f"{path_prefix}/{network_name}/mean.pt")
        torch.save(logvar, f"{path_prefix}/{network_name}/logvar.pt")

    if (type(features)) == np.ndarray:
        np.save(f"{path_prefix}/{network_name}/features.npy", features)
        np.save(f"{path_prefix}/{network_name}/mean.npy", mean)
        np.save(f"{path_prefix}/{network_name}/logvar.npy", logvar)


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
