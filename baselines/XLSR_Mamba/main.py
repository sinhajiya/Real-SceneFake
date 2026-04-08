import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import get_loader, OurEvalDataset
from model import Model
from utils import reproducibility
# from utils import read_metadata
import numpy as np
import csv
from datetime import datetime
from eer_calc import *
from pathlib import Path

def train_epoch(train_loader, model, lr,optim, device,weights):
    num_total = 0.0
    model.train()
    criterion = nn.CrossEntropyLoss(weight=weights)
    num_batch = len(train_loader)
    i=0
    pbar = tqdm(train_loader, total=num_batch)
    for batch_x, batch_y in pbar:    
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)     
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        i=i+1

    sys.stdout.flush()
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XLSR-Mamba')
    # Dataset
    parser.add_argument('--protocol_path', type=str, required=True, help='dir with protocols')
    parser.add_argument('--dataset', type=str, default='combined')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('--scenefake_eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('--model_path', type=str, default='/models/xlsr-mamba-scenefake-2026-03-31_15-18-32/best_dev.pth', help='model path')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size of the model')

    parser.add_argument('--num_encoders', type=int, default=12, metavar='N',
                    help='number of encoders of the mamba blocks')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    
    #Train
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)

    ##===================================================Rawboost data augmentation ======================================================================#
    parser.add_argument('--algo', type=int, default=0, 
                    help='Rawboost algos discriptions. (3 for DF, 5 for LA and ITW) 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    
    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')
    parser.add_argument(
    '--sf_protocols',
    nargs='+',
    default=[
        "/SceneFakeDataset/protocols/test_fold_1.txt",
        "/SceneFakeDataset/protocols/test_fold_2.txt",
        "/SceneFakeDataset/protocols/test_fold_3.txt",
        "/SceneFakeDataset/protocols/test_fold_4.txt",
        "/SceneFakeDataset/protocols/test_fold_5.txt",
        "/SceneFakeDataset/protocols/test_fold_6.txt",
        "/SceneFakeDataset/protocols/test_fold_7.txt",
        "/SceneFakeDataset/protocols/test_fold_8.txt",
        "/SceneFakeDataset/protocols/test_fold_9.txt",
        "/SceneFakeDataset/protocols/test_fold_10.txt",
        "/SceneFakeDataset/protocols/test_fold_11.txt",
        "/SceneFakeDataset/protocols/test_fold_12.txt",
    ],
    help='List of protocol files for SceneFake folds'
)
    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
	
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    print(args)
   
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    n_mejores=args.n_mejores_loss

    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'
    #database
    prefix      = args.dataset
    rn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #define model saving path
    model_tag = f'xlsr-mamba-{prefix}-{rn}'
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False

    model = model.to(device)

    if args.eval:
        model_root = os.path.dirname(os.path.dirname(args.model_path))
        metrics_dir = os.path.join(model_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        print("๋࣭ ⭑✮💻₊ ⊹ Evaluation mode ๋࣭ ⭑✮💻₊ ⊹ ")

        model = Model(args, device).to(device)

        model.load_state_dict(
            torch.load(args.model_path, map_location=device)
        )

        print("Loaded model:", args.model_path)

        _, _, seen_loader, unseen_loader, _ = get_loader(args.seed, args)


        if not args.scenefake_eval:
            model.eval()
            with torch.no_grad():
                metrics_dir1 = os.path.join(metrics_dir, "seen")
                seen_eer, _, _, _ = evaluate_eer_utterance(args,seen_loader, model, device, metrics_dir1)
                print("Seen EER:", seen_eer)

                metrics_dir1 = os.path.join(metrics_dir, "segs_seen")
                seen_eer, _, _, _ = evaluate_eer_segments(args,seen_loader, model, device, metrics_dir1)
                print("Seen EER segment:", seen_eer)


                if unseen_loader is not None:
                    metrics_dir1 = os.path.join(metrics_dir, "unseen")
                    unseen_eer, _, _, _ = evaluate_eer_utterance(args,unseen_loader, model, device, metrics_dir1)
                    print("Unseen EER:", unseen_eer)
                    metrics_dir1 = os.path.join(metrics_dir, "unseen_segs")
                    unseen_eer, _, _, _ = evaluate_eer_segments(args,unseen_loader, model, device, metrics_dir1)
                    print("Unseen EER segment:", unseen_eer)
            sys.exit(0)
        
        else:

            print(" scenefake eval")
            assert args.sf_protocols is not None, "Provide --sf_protocols"
            
            model.eval()

            all_eers = []
            all_reports = []

            with torch.no_grad():
                for i, loader in enumerate(seen_loader):

                
                    print(f"\nFOLD {i+1}")
                    metrics_dir1 = os.path.join(metrics_dir, f"sf/folf_{i+1}")

                    eer, _, _, report = evaluate_eer_utterance(
                        args, loader, model, device,metrics_dir1
                    )
                    all_eers.append(eer)
                    all_reports.append(report)

            print("\n final---- avg ")

            if len(all_eers) > 0:
                print(f"Mean EER: {np.mean(all_eers):.4f}")
                print(f"Std EER : {np.std(all_eers):.4f}")
            else:
                print("No valid folds for EER.")

            avg_report = average_classification_reports(all_reports)

            print_report(avg_report)

            sys.exit(0)


    #set Adam optim
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    model_save_path = os.path.join('models', model_tag)
    
    print('Model tag: '+ model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)

    metrics_dir = os.path.join(model_save_path, "val_metrics")

    # os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)


    trn_loader, val_loader, seen_loader, unseen_loader, class_weights = get_loader(args.seed, args)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    best_dev_eer = float('inf')
    best_epoch = -1
    early_stop_counter = 0
    MIN_DELTA = 0.0
    EARLY_STOP_PATIENCE = 5
    print("training 🦁")
    num_epochs = args.num_epochs
    not_improving=0
    epoch=0
    bests=np.ones(n_mejores,dtype=float)*float('inf')
    best_loss=float('inf')
    if args.train:
        for i in range(n_mejores):
            np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
        while not_improving<args.num_epochs:
            print('######## Epoch {} ########'.format(epoch))
            train_epoch(trn_loader, model, args.lr, optim, device, class_weights)
            metrics_dir_epoch = os.path.join(metrics_dir, f"val_{epoch}")
            dev_eer, _, _, _ = evaluate_eer_utterance(
                args,
                val_loader,
                model,
                device,
                metrics_dir_epoch
            )
            improved = dev_eer < (best_dev_eer - MIN_DELTA)
            if improved:
                

                best_dev_eer = dev_eer
                best_epoch = epoch
                early_stop_counter = 0

                print(f" New best EER at epoch {epoch}")

                torch.save(
                    model.state_dict(),
                    os.path.join(model_save_path, "best_dev.pth")
                )
        
            else:
                early_stop_counter += 1
                print(f"No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

            for i in range(n_mejores):
                if bests[i] > dev_eer:   # lower EER is better
                    for t in range(n_mejores - 1, i, -1):
                        bests[t] = bests[t - 1]

                        os.system(
                            'mv {}/best_{}.pth {}/best_{}.pth'.format(
                                best_save_path, t - 1, best_save_path, t
                            )
                        )

                    bests[i] = dev_eer

                    torch.save(
                        model.state_dict(),
                        os.path.join(best_save_path, f'best_{i}.pth')
                    )
                    break

            print('\n{} - {}'.format(epoch, dev_eer))
            print('n-best loss:', bests)
            epoch+=1
            if epoch>74:
                break
        print('Total epochs: ' + str(epoch) +'\n')

