from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from evaluation import compute_eer 
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from evaluation import compute_det_curve

def evaluate_eer_utterance(conf,loader, model, device, metrics_path):
    print("this time i am doing it ki for each segment in the utterance, first find the logits of them (_x2) then staack the logits, find softmaax, get the dcore, witj these scores i proceed to find the eer using det curve.")
    os.makedirs(metrics_path, exist_ok=True)
    model.eval()

    logits_dict = defaultdict(list)
    labels_dict = {}

    with torch.no_grad():
        for batch_x, batch_y, batch_path in tqdm(loader, desc="EVAL LOGS", leave=False):

            batch_x = batch_x.to(device)
            if conf['model']=='aasist':
                _, logits = model(batch_x, Freq_aug=False)

            else:
                logits = model(batch_x)

            for logit, label, path in zip(logits, batch_y, batch_path):
                logits_dict[path].append(logit.cpu())
                labels_dict[path] = label.item()

    scores = []
    targets = []

    for path in logits_dict:
        stacked = torch.stack(logits_dict[path])  

        mean_logits = stacked.mean(dim=0)
        score = torch.softmax(mean_logits, dim=0)[1].item()

        scores.append(score)
        targets.append(labels_dict[path])

    scores = np.array(scores)
    targets = np.array(targets)

    real_scores = scores[targets == 0]
    fake_scores = scores[targets == 1]

    # eer, threshold = compute_eer(real_scores, fake_scores)
    eer, threshold = compute_eer(fake_scores, real_scores)
    eer = eer * 100

    pred_labels = (scores >= threshold).astype(int)

    cm = confusion_matrix(targets, pred_labels)

    report = classification_report(
        targets,
        pred_labels,
        target_names=["real", "fake"], 
        output_dict=True
    )

    print("Num utterances:", len(scores))
    print("Num real:", (targets == 0).sum())
    print("Num fake:", (targets == 1).sum())

    print("\nEER: {:.4f}".format(eer))
    print("Threshold (EER): {:.6f}".format(threshold))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)


    frr, far, thresholds_det = compute_det_curve(fake_scores, real_scores)

  

    plt.figure()
    plt.plot(far, frr)
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title("DET Curve")
    plt.grid()

    plt.scatter([eer/100], [eer/100]) 
    
    plt.savefig(f"{metrics_path}/det_curve.png")
    plt.close()
    np.savez(
        f"{metrics_path}/det_data.npz",
        frr=frr,
        far=far,
        thresholds=thresholds_det
    )
    return eer, threshold, cm, report

def average_classification_reports(reports):
    avg = {}

    for key in reports[0].keys():

        if isinstance(reports[0][key], dict):
            avg[key] = {}

            for metric in reports[0][key]:
                values = [r[key][metric] for r in reports]
                avg[key][metric] = np.mean(values)

        else:
            avg[key] = np.mean([r[key] for r in reports])

    return avg

def print_report(report):

    for cls in ["real", "fake"]:
        print(f"\nClass: {cls}")
        print(f"  Precision: {report[cls]['precision']:.4f}")
        print(f"  Recall   : {report[cls]['recall']:.4f}")
        print(f"  F1-score : {report[cls]['f1-score']:.4f}")

    print("\nOverall:")
    print(f"  Accuracy: {report['accuracy']:.4f}")



def evaluate_eer_segments(args, loader, model, device, metrics_path):
    os.makedirs(metrics_path, exist_ok=True)
    model.eval()

    scores = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y, _ in tqdm(loader, desc="SEGMENT EVAL", leave=False):

            batch_x = batch_x.to(device)
            logits = model(batch_x)                  # [B, 2]
            probs = torch.softmax(logits, dim=1)     # [B, 2]

            batch_scores = probs[:, 1].cpu().numpy() # fake prob
            batch_targets = batch_y.cpu().numpy()

            scores.extend(batch_scores)
            targets.extend(batch_targets)

    scores = np.array(scores)
    targets = np.array(targets)

    real_scores = scores[targets == 0]
    fake_scores = scores[targets == 1]

    
    eer, threshold = compute_eer(fake_scores, real_scores)
    eer = eer * 100

    pred_labels = (scores >= threshold).astype(int)

    cm = confusion_matrix(targets, pred_labels)

    report = classification_report(
        targets,
        pred_labels,
        target_names=["real", "fake"],
        output_dict=True
    )

    print("\n[SEGMENT-LEVEL RESULTS]")
    print("Num segments:", len(scores))
    print("EER: {:.4f}".format(eer))
    print("Threshold:", threshold)
    print("Confusion Matrix:\n", cm)

    print_report(report)
    frr, far, thresholds_det = compute_det_curve(real_scores, fake_scores)

    plt.figure()
    plt.plot(far, frr)
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET Curve (Segment-level)")
    plt.grid()

    idx = np.argmin(np.abs(far - frr))
    plt.scatter(far[idx], frr[idx])

    plt.savefig(f"{metrics_path}/det_curve_segment.png")
    plt.close()

    np.savez(
        f"{metrics_path}/det_segment.npz",
        frr=frr,
        far=far,
        thresholds=thresholds_det
    )

    return eer, threshold, cm, report