import image_lddmm_tensorflow.lddmm as lddmm
import pickle
import numpy as np
import os
from scipy import misc
import matplotlib.pyplot as plt
import sys


# sys.setdefaultencoding("utf-8")

pkl_dir = '/home/samik/mnt/bnb/mnt/grid/mitra/hpc/home/data/hliu/public/STP_registration_pipeline/data/transfer_para/180816/'
pkl_file = 'transfer_para'

pkl_full_file = pkl_dir + pkl_file + '.pickle'

img_dir = '/home/samik/mnt/bnb/mnt/grid/mitra/hpc/home/data/banerjee/180816_JH_WG_Tle4LSLFlpNPCfa_female/Prelim_processDetection/outPath/180816_JH_WG_Tle4LSLFlpNPCfa_female/dsImg/setAvg/'


def main():
    # for item in read_from_pickle(pkl_full_file):
    #         print(repr(item))
    with open(pkl_full_file, 'rb') as f:
        out = pickle.load(f)
    for file in os.listdir(img_dir):
        if file.endswith(".jp2"):
            img_data = misc.imread(file)
            plt.imshow(img_data)
            nx = img_data.shape
            dx = np.array([10, 10, 10])

            x = [np.arange(nxi) * dxi - np.mean(np.arange(nxi) * dxi) for nxi, dxi in zip(nx, dx)]
            x0, x1, x2 = x


            transformed_data = lddmm.transform_data(x0, x1, x2, img_data,
                                                    out['Aphi0'], out['Aphi1'], out['Aphi2'])




def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

if __name__ == '__main__':
    main()