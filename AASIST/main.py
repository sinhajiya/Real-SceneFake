"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import datetime
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from tqdm import tqdm
from data_utils import (protocol_reader, OurTrainDataset, OurEvalDataset)
from eer_calc import *
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
def main(args: argparse.Namespace) -> None:

    print("Loading config from {}".format(args.config))

    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    print("Checking device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cpu":
        raise ValueError("GPU required")

    print("Experiment config:")
    for key, val in config.items():
        print("{}: {}".format(key, val))

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    if "freq_aug" not in config:
        config["freq_aug"] = "False"   

    set_seed(args.seed, config)

    if args.eval:
        
        model_root = os.path.dirname(os.path.dirname(config['model_path']))
        metrics_dir = os.path.join(model_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)


        print("๋࣭ ⭑✮💻₊ ⊹ Evaluation mode ๋࣭ ⭑✮💻₊ ⊹ ")

        model = get_model(model_config, device)

        model.load_state_dict(
            torch.load(config["model_path"], map_location=device)
        )

        print("Loaded model:", config["model_path"])

        _, _, seen_loader, unseen_loader, _ = get_loader(
            args.seed,
            config, args
        )

        if not config['scenefake-eval']:
            model.eval()
            with torch.no_grad():
                metrics_dir1 = os.path.join(metrics_dir, "seen")
                seen_eer, _, _, _ = evaluate_eer_utterance(seen_loader, model, device, metrics_dir1)
                print("Seen EER:", seen_eer)

                if unseen_loader is not None:
                    metrics_dir1 = os.path.join(metrics_dir, "unseen")

                    unseen_eer, _, _, _ = evaluate_eer_utterance(unseen_loader, model, device, metrics_dir1)
                    print("Unseen EER:", unseen_eer)
            sys.exit(0)
        
        else:

            print("ffaaaaaaaah scenefake evallll faaaaah")
            model.eval()

            all_eers = []
            all_reports = []

            with torch.no_grad():
                for i, loader in enumerate(seen_loader):

                    print(f"\nFOLD {i+1}")
                    metrics_dir1 = os.path.join(metrics_dir, f"sf/folf_{i+1}")

                    eer, _, _, report = evaluate_eer_utterance(
                        loader, model, device,metrics_dir1
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

    output_dir = Path(args.output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_tag = output_dir / f"{config['dataset_name']}_{config['suffix']}_{timestamp}"
    model_save_path = model_tag / "weights"
    model_save_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = model_tag / "val_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    copy(args.config, model_tag / "config.json")
    
    trn_loader, val_loader, seen_loader, unseen_loader, class_weights = get_loader(
        args.seed,
        config, args
    )

    model = get_model(model_config, device)

    optim_config["steps_per_epoch"] = len(trn_loader)

    optimizer, scheduler = create_optimizer(
        model.parameters(),
        optim_config
    )

    optimizer_swa = SWA(optimizer)

    EARLY_STOP_PATIENCE = 5
    MIN_DELTA = 1e-4

    best_dev_eer = float("inf")
    early_stop_counter = 0
    best_epoch = -1

    n_swa_update = 0

    for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs"):

        print(f"\nStart training epoch {epoch + 1}")

        model.train()
        running_loss = train_epoch(
            trn_loader,
            model,
            optimizer,
            device,
            scheduler,
            config, class_weights
        )

        
        print("𓂃˖˳·˖ ִֶָ ⋆🌷͙⋆ ִֶָ˖·˳˖𓂃 ִֶָ validation... 𓂃˖˳·˖ ִֶָ ⋆🌷͙⋆ ִֶָ˖·˳˖𓂃 ִֶָ")
        model.eval()
        metrics_dir1 = os.path.join(metrics_dir, f"val_{epoch+1}")

        with torch.no_grad():
            dev_eer, _, _,_ = evaluate_eer_utterance(
                val_loader,
                model,
                device, metrics_dir1
            )

        print(f"Loss:{running_loss:.5f}, dev_eer:{dev_eer:.4f}")

    
        if dev_eer < best_dev_eer - MIN_DELTA:

            print("Best model at epoch", epoch)

            best_dev_eer = dev_eer
            best_epoch = epoch
            early_stop_counter = 0

            torch.save(
                model.state_dict(),
                model_save_path / "best_dev.pth"
            )

        else:
            early_stop_counter += 1
            print(f"No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        if epoch > 0.75 * config["num_epochs"]:
            optimizer_swa.update_swa()
            n_swa_update += 1

    print("\nTraining finished")
    print(f"Best epoch: {best_epoch}, Best Dev EER: {best_dev_eer:.4f}")

    model.load_state_dict(
        torch.load(model_save_path / "best_dev.pth")
    )

    if n_swa_update > 0:
        print("\nApplying SWA...")
        optimizer_swa.swap_swa_sgd()

        optimizer_swa.bn_update(
            trn_loader,
            model,
            device=device
        )

    

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_loader(seed: int, config: dict, args: argparse.Namespace):

    gen = torch.Generator()
    gen.manual_seed(seed)
    pp = Path(config["protocol_path"])
    train_protocol = pp / "train.txt"
    val_protocol = pp / "val.txt"

    

    if not args.eval:
        train_labels, train_files = protocol_reader(train_protocol)
        val_labels, val_files = protocol_reader(val_protocol)

        print("train files:", len(train_files))
        print("validation files:", len(val_files))
        print(type(train_labels))
        labels_array = np.array(list(train_labels.values()))
        class_counts = np.bincount(labels_array)
        print("Training class counts:", class_counts)
        total = len(labels_array)
        class_weights = total / (len(class_counts) * class_counts)

        print("Class weights:", class_weights)

        # print(train_labels.head())
        # print(dict(itertools.islice(train_labels.items(), 2)))

        # class_counts = np.bincount(train_labels)
        # print("Training class counts:", class_counts)

        # total = len(train_labels)
        # class_weights = total / (len(class_counts) * class_counts)

        # print("Class weights:", class_weights)

        train_set = OurTrainDataset(
            file_list=train_files,
            labels=train_labels,
            augmentations = config.get("augmentations", [])
        
        )

        trn_loader = DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=config.get("num_workers", 4),
            worker_init_fn=seed_worker,
            generator=gen
        )

        val_set = OurEvalDataset(
            file_list=val_files,
            labels=val_labels
        )

        val_loader = DataLoader(
            val_set,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=config.get("num_workers", 4)

        )
    else:
        # trn_loader, val_loader = None, None
        print("evalllll loaderrrrrr")

        if not config["scenefake-eval"]:
            print("not sf eval worked!")
            unseen_protocol = pp / "unseen_test.txt"
            seen_protocol = pp/ "seen_test.txt"

            seen_labels, seen_files = protocol_reader(seen_protocol)
            print("seen test files:", len(seen_files))

            seen_set = OurEvalDataset(
                file_list=seen_files,
                labels=seen_labels
            )
            seen_loader = DataLoader(
                seen_set,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=config.get("num_workers", 4)
            )

            if os.path.exists(unseen_protocol):
                unseen_labels, unseen_files = protocol_reader(unseen_protocol)

                print("unseen test files:", len(unseen_files))

                unseen_set = OurEvalDataset(
                    file_list=unseen_files,
                    labels=unseen_labels
                )

                unseen_loader = DataLoader(
                    unseen_set,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=config.get("num_workers", 4)
                )
            else:
                unseen_loader = None

            return None, None, seen_loader, unseen_loader, None
        
        else:
            print("main  sf eval and protocols wale tk pahcuh gya huuuu")
            test_protocols = config['sf-eval-protocols']
            seen_loader = []
            for num, t in enumerate(test_protocols):
                seen_labels, seen_files = protocol_reader(t)
                print("test files in fold :", num+1, " is ", len(seen_files))
                seen_set = OurEvalDataset(
                file_list=seen_files,
                labels=seen_labels
            )
                seen_loader.append(DataLoader(
                seen_set,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=config.get("num_workers", 4)
            )      )
                
            return None, None, seen_loader, None, None



    return trn_loader, val_loader, None, None, class_weights


def train_epoch(
    trn_loader: DataLoader,
    model, 
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
    class_weights=None):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader, desc="TRAINING LOGS.. 𐙚⋆°｡⋆♡", leave=False):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
