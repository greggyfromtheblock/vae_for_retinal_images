#!/home/zelhar/miniconda3/envs/test/bin/python
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import PIL as pil
from torchvision import transforms, utils
import argparse

parser = argparse.ArgumentParser(description='Database Image Size Scanner')
parser.add_argument('db', type=str, default=None, metavar='data_file',
                    help="""The path to the excel database which contains
                    metainformation about the imgages""")
parser.add_argument('dir', type=str, default=None,
        metavar='image_dir',
                    help="""The path to the directory which contains
                    the imgages""")
parser.add_argument('output', type=str, default=None,
        metavar='output',
                    help="""The name of the output database which contains
                    the extra information about the imgages""")
args = parser.parse_args()

#print(args, args.db, args.dir, args.output)

# Correcting the possibly missing slash.
#It's lame but works for this little purpose
if args.dir[-1] != '/':
    args.dir += '/'

db = 'odir/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
imdir = 'odir/ODIR-5K_Training_Dataset/'
out = 'odir/output.tsv'

imdir = args.dir
db = args.db
out = args.output

df = pd.read_excel(db)

#test = 'odir/ODIR-5K_Training_Dataset/7_left.jpg'
#simg = io.imread(test)
#img = pil.Image.open(test)


df['Left-Width'] = int(0)
df['Left-Height'] = int(0)
df['Right-Width'] = int(0)
df['Right-Height'] = int(0)


x = np.zeros_like(df['ID']).astype('int')
y = np.zeros_like(df['ID']).astype('int')
z = np.zeros_like(df['ID']).astype('int')
w = np.zeros_like(df['ID']).astype('int')

i=0
for row in df['Left-Fundus']:
    #print(df['Left-Fundus'])
    s = imdir + row
    t = imdir + row.replace("left", "right")
    img = pil.Image.open(s)
    x[i] = img.width
    y[i] = img.height
    img.close
    img = pil.Image.open(t)
    z[i] = img.width
    w[i] = img.height
    img.close
    i+=1

(x != z).sum()
(y != w).sum()

df['Left-Width'] = x
df['Left-Height'] = y
df['Right-Width'] = z
df['Right-Height'] = w

x/y
z/w

min(x)
min(y)
print('minimal/maximal size (width-height):',
        (min(x), min(y)), 
        (max(x),max(y)))
print('minimal/maximal aspect ration (width/height):',
        min(x/y), max(x/y)) 

print("saving the extended database to: ", out)
df.to_csv(out, index=False, sep='\t')

