import h5py
import os
import torch
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from common.models.backbone import clip

def extract_feature(clip_variant, device):
    image_dir = '/sdc1/songcl/imagecaptioning/data/%s2014'
    base_path = '/sdc1/songcl/imagecaptioning/data/annotations/captions_%s2014.json'

    save_path = os.path.join('/sdc1/songcl/imagecaptioning/data/', 'COCO2014_%s_A100_produce.hdf5' % clip_variant)
    f = h5py.File(save_path, mode='w')

    clip_model, transform = clip.load(clip_variant, device=device ,jit=False)
    image_model = clip_model.visual.to(device).eval()
    image_model.forward = image_model.intermediate_features
    
    for split in ['train', 'val']:
        ann_path = base_path % split
        coco = COCO(ann_path)

        with torch.no_grad():
            for img_id, img in tqdm(coco.imgs.items(), split):
                image_path = os.path.join(image_dir % split ,img['file_name'])

                image = Image.open(image_path).convert('RGB')
                image = transform(image)

                image = image.to(device).unsqueeze(0)
                gird, x = image_model.forward2(image)
                
                gird = gird.squeeze(0).cpu().numpy()
                x = x.squeeze(0).cpu().numpy()
                # print(gird.shape)
                # pp
                f.create_dataset('%s_features' % img_id, data=gird)
                f.create_dataset('%s_global' % img_id, data=x)

    f.close()

if __name__=='__main__':
    # clip_variant = 'RN50x4'#[81 2560]

    # clip_variant = 'RN50x16'#[144 3072]
    # clip_variant = 'RN50x64' #[196 4096]
    # clip_variant = 'ViT-B-32'#[49 768]
    # clip_variant = 'ViT-B-16'#[196 768]
    clip_variant = 'ViT-L-14'#[256 1024]
        # clip_variant ='ViT-L-14@336px'#[576, 1024]
    device = 'cuda:0'
    extract_feature(clip_variant, device)
