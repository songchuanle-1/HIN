import argparse
from common.data.field import DualImageField, TextField
# from common.utils.utils_online import create_dataset
from common.data.dataset_online import OnlineTestDataset
from models_of import build_encoder, build_decoder, Transformer, TransformerEnsemble
from common.utils.utils import setup_seed
import torch
import os
from shutil import copyfile
from common.data import DataLoader
from common.data.field import RawField
from common.utils import  evaluate_metrics_test_online
from common.utils.utils import setup_seed
import json
import pickle


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
    parser.add_argument('--resume_best', action='store_true')



    parser.add_argument('--clip_path', type=str,
                         default='/speedup_data/slurm_data/songcl_data_place/DFT_data/features/coco2014/COCO2014_ViT-L-14_A100_produce.hdf5')
    parser.add_argument('--vinvl_path', type=str, 
                        default='/speedup_data/slurm_data/songcl_data_place/DFT_data/features/COCO2014_VinVL.hdf5')
    parser.add_argument('--ann_path', type=str, default='outputs/captions_val2014.json')    
    parser.add_argument('--output_path', type=str, default='outputs/captions_val2014_SD_results.json')  



    parser.add_argument('--image_folder', type=str, 
                        default='/speedup_data/slurm_data/songcl_data_place/DFT_data/features/coco2014/')
    parser.add_argument('--annotation_folder', type=str,
                         default='/speedup_data/slurm_data/songcl_data_place/DFT/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print(args.output_path)
    # Pipeline for image features
    image_field = DualImageField(args.clip_path, args.vinvl_path, max_detections=50)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    vocab_path = 'cache/vocab.pkl'
    # if not os.path.isfile(vocab_path):
    #     print("Building vocabulary")
    #     text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
    #     pickle.dump(text_field.vocab, open(vocab_path, 'wb'))
    # else:
    text_field.vocab = pickle.load(open(vocab_path, 'rb'))

    encoder = build_encoder(args.N_enc, device=args.device)
    decoder = build_decoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)
    ### ensemble

    model_path_list = [
        'coco/checkpoints/paper_weights/decoder_cca_multi_feature_gate1_encoder_gce_query2att_exchange_add_mask_output_threetoken_middle_feature2_noshortcut_for_decoder_rl_best_test.pth',
        'coco/checkpoints/paper_weights/decoder_cca_multi_feature_gate1_decoder_cca_multi_feature_gate1_encoder_gce_query2att_exchange_add_mask_output_threetoken_middle_feature2_noshortcut_for_decoder_seed520_rl_best_test.pth',
        # 'coco/checkpoints/paper_weights_bs100/decoder_cca_multi_feature_gate1_decoder_cca_multi_feature_gate1_encoder_gce_query2att_exchange_add_mask_output_threetoken_middle_feature2_noshortcut_for_decoder_seed521_best_test.pth',
        'coco/checkpoints/paper_weights/decoder_cca_multi_feature_gate1_decoder_cca_multi_feature_gate1_encoder_gce_query2att_exchange_add_mask_output_threetoken_middle_feature2_noshortcut_for_decoder_seed170521_rl_best_test.pth',
        'coco/checkpoints/paper_weights/decoder_cca_multi_feature_gate1_decoder_cca_multi_feature_gate1_encoder_gce_query2att_exchange_add_mask_output_threetoken_middle_feature2_noshortcut_for_decoder_seed111_rl_best_test.pth',
                #    'coco/checkpoints/paper_weights_bs100/decoder_cca_multi_feature_gate1_decoder_cca_multi_feature_gate1_encoder_gce_query2att_exchange_add_mask_output_threetoken_middle_feature2_noshortcut_for_decoder_seed320_best_test.pth',
                #    
                    ]
    model_ensemble = TransformerEnsemble(model,model_path_list,device=args.device)
    test(args, model_ensemble, image_field, text_field)




# setup_seed(521)
def test(args, model, image_field, text_field):
          
    device = args.device
    output = args.output
    use_rl = args.use_rl

    # ann_path = 'annotations/image_info_test2014.json'
    # output_path = 'outputs/captions_test2014_SD_results.json'

    # # ann_path = 'outputs/captions_val2014.json'
    # # output_path = 'outputs/captions_val2014_SD_results.json'
    dataset_test = OnlineTestDataset(args.ann_path, {'image': image_field, 'text': RawField()})
    dataset = dataset_test.image_dictionary({'image': image_field, 'text': RawField()})
    
    train_batch_size = args.batch_size // 5 if use_rl else args.batch_size
    dict_dataloader_test = DataLoader(dataset, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)

    result = evaluate_metrics_test_online(model, dict_dataloader_test, text_field, -1, device,True, args)
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    with open(args.output_path, 'w') as fp:
        json.dump(result, fp)

    print("Online test result is over")
    

if __name__ == "__main__":
    
    args = parse_args()
    main(args)

    