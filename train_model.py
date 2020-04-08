"""
Trigger training here
"""
import os
import sys
from torchvision import datasets, transforms
import torch
from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset
from utils.utils import setup
import time
import argparse


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def add_slash(path):
    if path[-1] != '/':
        return path + "/"
    else:
        return(path)


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/training.py", config="config.json")
    print("FLAGS= ", FLAGS)

    imfolder = add_slash(FLAGS.input)
    network_name = FLAGS.network_name
    path_prefix = FLAGS.path_prefix

    network_dir = f'{path_prefix}/{network_name}/'

    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    print("\ninput dir: ", imfolder,
          "\ndevice: ", device)
    os.makedirs(network_dir, exist_ok=True)
    if FLAGS.network_name in os.listdir(network_dir):
        input1 = input("\nNetwork already exists. Are you sure to proceed? ([y]/n) ")
        if not input1 in ['y', 'yes']:
            sys.exit()

    print("\nLoad Data as Tensors...")
    img_dataset = datasets.ImageFolder(
        # Because dataloader asks for the parent directory
        os.path.dirname(os.path.dirname(imfolder)),
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    data = VAEDataset(img_dataset)

    encoder, decoder = Encoder(z=FLAGS.zdim), Decoder(z=FLAGS.zdim)

    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        path_prefix=path_prefix,
        network_name=network_name,
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

    # Save network
    # os.makedirs(network_dir)
    PATH = network_dir+f'{network_name}.pth'
    torch.save(trained_encoder.state_dict(), PATH)

