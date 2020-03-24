import pandas as pd
import numpy as np
import PIL as pil


db = 'odir/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
imdir = 'odir/ODIR-5K_Training_Dataset/'
out = 'odir/output.tsv'


def find_optimal_image_size_and_extend_db(db):
    df = pd.read_excel(db)

    df['Left-Width'] = int(0)
    df['Left-Height'] = int(0)
    df['Right-Width'] = int(0)
    df['Right-Height'] = int(0)

    x = np.zeros_like(df['ID']).astype('int')
    y = np.zeros_like(df['ID']).astype('int')
    z = np.zeros_like(df['ID']).astype('int')
    w = np.zeros_like(df['ID']).astype('int')

    min_x, min_y = float('inf'), float('inf')
    for i, row in enumerate(df['Left-Fundus']):
        s = imdir + row
        t = imdir + row.replace("left", "right")
        img = pil.Image.open(s)
        x[i] = img.width
        y[i] = img.height
        if img.width*img.height < min_x * min_y:
            min_x, min_y = img.width, img.height

        img.close
        img = pil.Image.open(t)
        z[i] = img.width
        w[i] = img.height
        img.close

    (x != z).sum()
    (y != w).sum()

    df['Left-Width'] = x
    df['Left-Height'] = y
    df['Right-Width'] = z
    df['Right-Height'] = w

    print('minimal/maximal size (width-height):',
            (min(x), min(y)),
            (max(x),max(y)))
    print('minimal/maximal aspect ration (width/height):',
            min(x/y), max(x/y))
    print('Total ration (left_width/left_height):',
            np.sum(x)/np.sum(y))
    print('\nMinimal image size: %ix%i\n' % (int(min_x), int(min_y)))

    print("saving the extended database to: ", out)
    df.to_csv(out, index=False, sep='\t')

    