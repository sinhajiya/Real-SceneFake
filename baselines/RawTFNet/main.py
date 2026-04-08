import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from datetime import datetime
import json
from pathlib import Path
from data_utils_SSL import *
import torch.nn.functional as F
import torchaudio.functional as AF
import soundfile as sf
from eer_calc import *
import random
from utils import *


def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def average_model(model, n_average_model, model_ID_to_average, best_save_path, model_folder_path):
    sd = None
    for ID in model_ID_to_average:
        model.load_state_dict(torch.load(os.path.join(model_folder_path, 'epoch_{}.pth'.format(ID))))
        print('Model loaded : {}'.format(os.path.join(model_folder_path, 'epoch_{}.pth'.format(ID))))
        if sd is None:
            sd = model.state_dict()
        else:
            sd2 = model.state_dict()
            for key in sd:
                sd[key] = (sd[key] + sd2[key])
    for key in sd:
        sd[key] = (sd[key]) / n_average_model
    model.load_state_dict(sd)
    torch.save(model.state_dict(), best_save_path)
    print('Model loaded average of {} best models in {}'.format(n_average_model, best_save_path))
def produce_evaluation_file(dataset, model, device, save_path, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=20)
    model.eval()
    for batch_x, utt_id in tqdm(data_loader):
        fname_list = []
        score_list = []
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
        ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sf-rawtfnet')
   
   
    # model
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument('--n_output_logits', type=int, default=2, help='number of output logits for the model, default is 2, following wav2vec2-AASIST repo')
    parser.add_argument('--date', type=str, default='unknown',
                        help='date')
    parser.add_argument('--model_name', type=str, default='rawtfnet' , choices=['rawtfnet','rawtfnet_small'], help='the type of the model, check from the choices')
    parser.add_argument('--protocols_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument(
    '--sf_eval_protocols',
    nargs='+',
    default=[
        "../../datasets/SF/protocols/test_fold_1.txt",
        "../../datasets/SF/protocols/test_fold_2.txt",
        "../../datasets/SF/protocols/test_fold_3.txt",
        "../../datasets/SF/protocols/test_fold_4.txt",
        "../../datasets/SF/protocols/test_fold_5.txt",
        "../../datasets/SF/protocols/test_fold_6.txt",
        "../../datasets/SF/protocols/test_fold_7.txt",
        "../../datasets/SF/protocols/test_fold_8.txt",
        "../../datasets/SF/protocols/test_fold_9.txt",
        "../../datasets/SF/protocols/test_fold_10.txt",
        "../../datasets/SF/protocols/test_fold_11.txt",
        "../../datasets/SF/protocols/test_fold_12.txt",
    ],
    help='List of protocol files for SceneFake folds'
)
    # Hyperparameters
    parser.add_argument('--num_average_model', type=int, default=1)
    parser.add_argument("--model_ID_to_average", type=int, nargs='+', default=[], help="Nes_ratio, from outer to inner")
    parser.add_argument("--model_folder_path", type=str, help="the path of the folder of saved checkpoints")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    # model name and path
    parser.add_argument('--model_path', type=str,
                        default='models/rawtfnet_combined_2026-04-02_03-58-05/best_dev.pth', help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode')
    
    parser.add_argument('--scenefake_eval', action='store_true', default=False, help='sf- eval mode-12-folds')
    
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False,
                        help='use cudnn-benchmark? (default false)')

    parser.add_argument('--algo', type=int, default=0,
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2), 9: parallel algo(1,2,3), 10: parallel algo(1,2) with possibility \
                          11: series algo (1+2+3) with possibility [default=0]')
    parser.add_argument('--LnL_ratio', type=float, default=1.0,
                    help='This is the possibility to activate LnL, which will only be used when algo>=10.')
    parser.add_argument('--ISD_ratio', type=float, default=1.0,
                    help='This is the possibility to activate ISD, which will only be used when algo>=10.')
    parser.add_argument('--SSI_ratio', type=float, default=1.0,
                    help='This is the possibility to activate SSI, which will only be used when algo>=11.')
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

    
    set_seed(args.seed, args)
    prefix = args.dataset
    rn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    if args.model_name == 'rawtfnet':
        from model_scripts.rawtfnet import RawTFNet as Model
    elif args.model_name == 'rawtfnet_small':
        from model_scripts.rawtfnet import RawTFNet_small as Model
    else:
        raise ValueError
    
    model = Model().to(device)

    num_params = sum([param.view(-1).size()[0] for param in model.parameters()])
   
    print('num_params:', num_params)

    if args.eval:
        model_root = os.path.dirname(os.path.dirname(args.model_path))
        metrics_dir = os.path.join(model_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        print("๋࣭ ⭑✮💻₊ ⊹ Evaluation mode ๋࣭ ⭑✮💻₊ ⊹ ")

        model = Model().to(device)

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

                metrics_dir1 = os.path.join(metrics_dir, "seen_segs")
                seen_eer, _, _, _ = evaluate_eer_segments(args,seen_loader, model, device, metrics_dir1)
                print("Seen EER segs:", seen_eer)

                if unseen_loader is not None:
                    metrics_dir1 = os.path.join(metrics_dir, "unseen")

                    unseen_eer, _, _, _ = evaluate_eer_utterance(args,unseen_loader, model, device, metrics_dir1)
                    print("Unseen EER:", unseen_eer)
                    metrics_dir1 = os.path.join(metrics_dir, "unseen_segs")

                    unseen_eer, _, _, _ = evaluate_eer_segments(args,unseen_loader, model, device, metrics_dir1)
                    print("Unseen EER segs:", unseen_eer)

            sys.exit(0)
        
        else:

            print(" scenefake eval")
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


    # define model saving path
    model_tag = f'{args.model_name}_{prefix}_{rn}'
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)


    # set model save directory
    if not os.path.exists(model_save_path) and not args.eval:
        os.mkdir(model_save_path)

    metrics_dir = Path(model_save_path) / "val_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)


    # set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load model or average checkpoints
    if args.num_average_model > 1:  # average checkpoints
        assert len(args.model_ID_to_average) == args.num_average_model, ('num_average_model is not equal to the number of model IDs provided in model_ID_to_average')
        model_path_to_test = args.model_folder_path + '/Averaged_Model_IDs'
        for item in args.model_ID_to_average:
            model_path_to_test += "_{}".format(item)
        model_path_to_test += ".pth"
        if os.path.exists(model_path_to_test):
            print(f"File '{model_path_to_test}' already exists. Model averaging operation skipped.")
        else:
            print(f"File '{model_path_to_test}' does not exist. Proceeding with the model averaging operation...")
            average_model(model=model, n_average_model=args.num_average_model, model_ID_to_average=args.model_ID_to_average,
                      best_save_path=model_path_to_test, model_folder_path=args.model_folder_path)
        model.load_state_dict(torch.load(model_path_to_test, map_location=device))
        print('Model loaded : {}'.format(model_path_to_test))


    print("starting training.. 🦁")
    trn_loader, val_loader, seen_loader, unseen_loader, class_weights = get_loader(args.seed, args)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
   
    num_epochs = args.num_epochs
    
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    loss_min_w_aug = 999
    loss_min_wo_aug = 999
    epoch_loss_w_aug = 0
    epoch_loss_wo_aug = 0
    loss_min = float('inf')
    best_epoch = 0
    EARLY_STOP_PATIENCE = 5
    MIN_DELTA = 1e-4
    best_dev_eer = float('inf')
    early_stop_counter = 0


    for epoch in range(1, num_epochs + 1):

        if epoch == 1:
            s = time.strftime("%a, %d %b %Y %I:%M:%S", time.gmtime())
            print("training started at:", s)

        st_t = time.time()

        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in tqdm(trn_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).long().to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        train_loss = total_loss / total_samples


        model.eval()

        metrics_dir_epoch = os.path.join(metrics_dir, f"val_{epoch}")

        with torch.no_grad():
            dev_eer, _, _, _ = evaluate_eer_utterance(
                args,
                val_loader,
                model,
                device,
                metrics_dir_epoch
            )

        print(f"[Epoch {epoch}] train_loss: {train_loss:.5f}, dev_eer: {dev_eer:.4f}")


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

        print(f"time: {time.time() - st_t:.2f}s\n")
