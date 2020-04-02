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


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/training.py", config="config.json")
    print("FLAGS= ", FLAGS)

    imfolder = os.path.abspath(FLAGS.input)
    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    print("\ninput dir: ", imfolder,
          "\ndevice: ", device)

    os.makedirs(FLAGS.path_prefix, exist_ok=True)
    if FLAGS.networkname in os.listdir(FLAGS.path_prefix):
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
        path_prefix=FLAGS.path_prefix,
        network_name=FLAGS.networkname,
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
    PATH = f'{FLAGS.path_prefix}/{FLAGS.networkname}/{FLAGS.networkname}.pth'
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
