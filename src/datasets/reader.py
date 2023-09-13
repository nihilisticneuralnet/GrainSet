import os
import pandas as pd
import random

__all__ = ['get_imglists']

def get_imglists(root, split='train', phase='train'):
    '''
    get all images path
    @param: 
        root : root path to dataset
        spilt: sub path to specific dataset folder
    '''

    grain = root.split('/')[-1]
    imgs, labels = [], []

    if split == 'train':
        split = phase

    with open(os.path.join('./runs/datalist', f'{grain}_{split}.txt')) as f:
        for line in f.readlines():
            line = line.replace('\n','')
            im_path, label = line.split()
            im_path = os.path.join(root, im_path)
            imgs.append(im_path)
            labels.append(int(label))
    length = len(imgs)
    print(f'* {split} : {length}')
    files = pd.DataFrame({'filename': imgs, 'label': labels})
    return files
