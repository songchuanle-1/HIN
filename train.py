import argparse
from common.data.field import DualImageField, TextField
from common.train import train
from common.utils.utils import create_dataset
from models_of import build_encoder, build_decoder, Transformer
from common.utils.utils import setup_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Dual Transformer_of')
  
    parser.add_argument('--output', type=str, default='paper_weights')
    parser.add_argument('--exp_name', type=str, default='default')  
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)

    parser.add_argument('--xe_base_lr', type=float, default=1e-4)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--clip_path', type=str, default='/sdc1/songcl/imagecaptioning/data/COCO2014_ViT-L-14_A100_produce.hdf5')

    parser.add_argument('--vinvl_path', type=str, default='/sdc1/songcl/imagecaptioning/data/COCO2014_VinVL.hdf5')
    parser.add_argument('--image_folder', type=str, default='/sdc1/songcl/imagecaptioning/data/')
    parser.add_argument('--annotation_folder', type=str, default='/sdc1/songcl/imagecaptioning/data/annotations')
    
    parser.add_argument('--vocab_path', type=str, default='/sdc1/songcl/imagecaptioning/HIN/code/HIN/cache/vocab.pkl')
    parser.add_argument('--cider_cache_path', type=str, default='/sdc1/songcl/imagecaptioning/HIN/code/HIN/cache/cider_cache.pkl')
    
   
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print('HIN Training')
    setup_seed(111)

    image_field = DualImageField(args.clip_path, args.vinvl_path, max_detections=50)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    datasets = create_dataset(args, image_field, text_field)

    encoder = build_encoder(args.N_enc, device=args.device)
    decoder = build_decoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)

    train(args, model, datasets, image_field, text_field)
    
if __name__ == "__main__":
    
    args = parse_args()
    main(args)
