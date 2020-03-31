import torch.nn as nn
from torch.utils.data import Dataset
from torchsupport.training.vae import VAETraining
from tensorboardX import SummaryWriter

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class VAEDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label = self.data[index]
    return (data,)

  def __len__(self):
    return len(self.data)

class Encoder(nn.Module):
  def __init__(self, z=32):
    super(Encoder, self).__init__()
    self.z = z
    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, z),
      nn.ReLU()
    )
    self.mean = nn.Linear(z, z)
    self.logvar = nn.Linear(z, z)

  def forward(self, inputs):
    inputs = inputs.view(inputs.size(0), -1)
    features = self.encoder(inputs)
    mean = self.mean(features)
    logvar = self.logvar(features)
    return features, mean, logvar

class Decoder(nn.Module):
  def __init__(self, z=32):
    super(Decoder, self).__init__()
    self.z = z
    self.decoder = nn.Sequential(
      nn.Linear(z, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 28 * 28)
    )

  def forward(self, sample):
    return self.decoder(sample).view(-1, 1, 28, 28)

class MNISTVAETraining(VAETraining):
  def __init__(self, encoder, decoder, data, network_name, path_prefix=".",
               **kwargs):
    super(MNISTVAETraining, self).__init__(
      encoder, decoder, data,
      **kwargs
    )
    self.checkpoint_path = f"{path_prefix}/{network_name}/{network_name}-checkpoint"
    self.writer = SummaryWriter(f"{path_prefix}/{network_name}/")
    self.epoch = None

  def run_networks(self, data, *args):
    if self.epoch != self.epoch_id:
      self.epoch = self.epoch_id
      print("%i-Epoch" % self.epoch_id)

    mean, logvar, reconstruction, data = super().run_networks(data, *args)
    self.writer.add_image("target", normalize(data[0]), self.step_id)
    self.writer.add_image("reconstruction", normalize(reconstruction[0].sigmoid()), self.step_id)
    return mean, logvar, reconstruction, data

