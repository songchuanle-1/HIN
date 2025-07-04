import h5py
import os
save_path = os.path.join('../../coco', 'COCO2014_RN50x4.hdf5')
f = h5py.File(save_path, mode='r')
print(f)