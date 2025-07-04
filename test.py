import argparse
from common.data.field import DualImageField, TextField
from common.utils.utils import create_dataset
from models_of import build_encoder, build_decoder, Transformer
from common.utils.utils import setup_seed
import torch
import os
from shutil import copyfile
from common.data import DataLoader
from common.data.field import RawField
from common.utils import  evaluate_metrics_test
from common.utils.utils import setup_seed



def parse_args():
    parser = argparse.ArgumentParser(description='Dual Transformer_of')
    parser.add_argument('--output', type=str, default='data/checkpoints')
    parser.add_argument('--exp_name', type=str, default='HIN')  
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)

    parser.add_argument('--xe_base_lr', type=float, default=1e-4)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--resume_best',type=bool, default= True)
    parser.add_argument('--clip_path', type=str,
                         default='/sdc1/songcl/imagecaptioning/data/COCO2014_ViT-L-14_A100_produce.hdf5')

    parser.add_argument('--vinvl_path', type=str, 
                        default='/sdc1/songcl/imagecaptioning/data/COCO2014_VinVL.hdf5')
    parser.add_argument('--image_folder', type=str, 
                        default='/sdc1/songcl/imagecaptioning/data/')
    parser.add_argument('--annotation_folder', type=str,
                         default='/sdc1/songcl/imagecaptioning/data/annotations')
    
    parser.add_argument('--vocab_path', type=str, default='/sdc1/songcl/imagecaptioning/HIN/code/HIN/cache/vocab.pkl')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print('HIN Testing')
    # Pipeline for image features
    image_field = DualImageField(args.clip_path, args.vinvl_path, max_detections=50)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',   remove_punctuation=True, nopoints=False)

    datasets = create_dataset(args, image_field, text_field)
    encoder = build_encoder(args.N_enc, device=args.device)
    decoder = build_decoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)
    setup_seed(123456)
    test(args, model, datasets, image_field, text_field)


# setup_seed(521)
def test(args, model, datasets, image_field, text_field):
          
    device = args.device
    output = args.output
    use_rl = args.use_rl

    _, _, test_dataset = datasets
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    train_batch_size = args.batch_size // 5 if use_rl else args.batch_size
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)


    best_saved = os.path.join(output, '%s_best_test.pth' % args.exp_name)
    
    if args.resume_best:
        if use_rl:
            best_saved = os.path.join(output, '%s_rl_best_test.pth' % args.exp_name)
        fname = best_saved

        if os.path.exists(fname):
            data = torch.load(fname,map_location=args.device)

            model.load_state_dict(data['state_dict'], strict=False)

            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))
        print('checkpoint path: ',best_saved)
    else:
        print('no_checkpoints')
        return 

    print("Testing starts")
    scores = evaluate_metrics_test(model, dict_dataloader_test, text_field, -1, device,True, args)
    print("Test scores", scores)
    

if __name__ == "__main__":
    
    args = parse_args()
    main(args)
