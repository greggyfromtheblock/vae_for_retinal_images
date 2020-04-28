import torch
from torchvision import datasets, transforms as T
import argparse
import sys

#imdir = 'smalldata'
#img_dataset = datasets.ImageFolder(
#        imdir, transform=T.Compose([T.CenterCrop(224), T.ToTensor()]),
#    )

#l = get_mean_std(imdir)

#img_dataset2 = datasets.ImageFolder(
#        imdir, transform=T.Compose([T.CenterCrop(224), T.ToTensor(),
#            T.Normalize(l[0],l[1])
#            ]),
#    )

def get_mean_std(dirpath):
    """returns the per channel mean and std of the images in the folder"""
    img_dataset = datasets.ImageFolder(
        dirpath, transform=T.Compose([T.CenterCrop(224), T.ToTensor()]),
    )
    means = []
    stds = []
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img,_ in img_dataset:
        #means.append(torch.mean(img, (1,2)))
        #stds.append(torch.std(img, (1,2)))
        mean += torch.mean(img, (1,2))
        std += torch.std(img, (1,2))
    #mean = torch.mean(torch.tensor(means))
    #std = torch.mean(torch.tensor(stds))
    mean /= len(img_dataset.imgs)
    std /= len(img_dataset.imgs)
    #print(float(mean), " , ", float(std))
    return (mean, std)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""""
        give it the image folder, it calculates the means 
        and the std values and prints them to stdout
        """
    )
    parser.add_argument(
        "input", type=str, default=None, help="""path for the image directory"""
    )
    args = parser.parse_args()
    imdir = args.input
    mean, std = get_mean_std(imdir)
    print(float(mean), " , ", float(std))
