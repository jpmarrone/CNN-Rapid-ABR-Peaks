import os
import time
import random
import pickle
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# %% 

script_folder = Path(__file__).resolve().parent
base_folder = script_folder.parents[1]

training_data_folder = base_folder/"data"/"training"
outputs_folder = base_folder/"outputs"

training_data_folder.mkdir(parents=True, exist_ok=True)
outputs_folder.mkdir(parents=True, exist_ok=True)

combined_data_path = training_data_folder/"ch1_cnn_trainingData_labeled_thresholded_min10_PORTABLE.pkl"

save_root = outputs_folder/"dataset_size_ablation_ch1"
save_root.mkdir(parents=True, exist_ok=True)



# %%

hp = {'num_conv_layers': 4,   
      'conv1_channels': 53,
      'conv2_channels': 90,
      'conv3_channels': 65,
      'kernel_size': 3,
      'stride': 1,
      'use_batch_norm': True,
      'slope': 0.08668347270261578,
      'pooling_type': 'max',
      'pooling_kernel_size': 2,
      'pooling_stride': 2,
      'dropout_conv': 0.08755577559324501,
      'dropout_fc': 0.8406188983306407,
      'num_fc_layers': 2,
      'fc_neurons': 50,
      'learning_rate': 0.0007824778017587336,
      'weight_decay': 0.04969917519136381,
      'batch_size': 128,
      'epochs': 250}


strat = "freq_percent"
stimuli = ["click", "4k", "8k", "10k", "16k"]
num_targets = 4
k = 10
time_per_index = 0.04095997267759563


# %% 


class peaksDataset(Dataset):
    #inputs primary and reference singals and ground truth labels
    def __init__(self, abr_prim, abr_ref, peaks):
        abr_prim = np.stack(abr_prim).astype(np.float32)
        abr_ref = np.stack(abr_ref).astype(np.float32)
        peaks = np.asarray(peaks, dtype=np.int64)
        
        abr_prim = torch.from_numpy(abr_prim).unsqueeze(1)
        abr_ref = torch.from_numpy(abr_ref).unsqueeze(1)

        self.signals = torch.cat((abr_prim, abr_ref), dim=1)
        self.targets = torch.from_numpy(peaks)

    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.signals[idx], self.targets[idx]


class peaksCNN(nn.Module):
    def __init__(self, signal_length, hp, num_classes, num_targets):
        super().__init__()

        slope = hp["slope"]
        kernel_size = hp["kernel_size"]
        
        #add zero padding to each side of singal before convolution 
            #without padding it cannot center the kernal on the first or last point...and the signal will shrink
        padding = kernel_size // 2

        self.num_classes = num_classes
        self.num_targets = num_targets

        self.conv = nn.Sequential(self.conv_block(2, hp["conv1_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv1_channels"], hp["conv2_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv2_channels"], hp["conv3_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv3_channels"], hp["conv3_channels"], kernel_size, padding, hp, slope))

        with torch.no_grad():
            dummy = torch.zeros(1, 2, signal_length)
            flattened_size = self.conv(dummy).view(1, -1).size(1)

        #creates the fully connected part of the cnn
        self.fc = nn.Sequential(nn.Linear(flattened_size, hp["fc_neurons"]), nn.LeakyReLU(negative_slope=slope),
                                nn.Dropout(hp["dropout_fc"]), nn.Linear(hp["fc_neurons"], num_targets * num_classes))

    #builds one convolutional block
    def conv_block(self, in_channels, out_channels, kernel_size, padding, hp, slope):
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=hp["stride"], padding=padding)]
        
        if hp["use_batch_norm"]:
            layers.append(nn.BatchNorm1d(out_channels))
            
        layers.append(nn.LeakyReLU(negative_slope=slope))

        if hp["pooling_type"] == "max":
            layers.append(nn.MaxPool1d(kernel_size=hp["pooling_kernel_size"], stride=hp["pooling_stride"]))
        else:
            layers.append(nn.AvgPool1d(kernel_size=hp["pooling_kernel_size"], stride=hp["pooling_stride"]))

        if hp["dropout_conv"] > 0:
            layers.append(nn.Dropout(hp["dropout_conv"]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), self.num_targets, self.num_classes)


def train_model_epoch(model, train_loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0

    for signals, targets in train_loader:
        signals = signals.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * signals.size(0)
    return total_loss / len(train_loader.dataset)


def adjust_prediction(signal, pred_index, mode, window, lower_bound, upper_bound):
    start = max(1, pred_index - window)
    end = min(len(signal) - 1, pred_index + window + 1)

    #if initial prediction is already the right local extrema, leave it alone
    if pred_index > 0 and pred_index < len(signal) - 1:
        is_peak = signal[pred_index] > signal[pred_index - 1] and signal[pred_index] > signal[pred_index + 1]
        is_trough = signal[pred_index] < signal[pred_index - 1] and signal[pred_index] < signal[pred_index + 1]
                
        within_lower_bound = lower_bound is None or pred_index > lower_bound
        within_upper_bound = upper_bound is None or pred_index < upper_bound
        
        if within_lower_bound and within_upper_bound:
            if mode == "peak" and is_peak:
                return pred_index
            if mode == "trough" and is_trough:
                return pred_index
                
    candidates = []

    #if its not already an extrema it goes through the extrema check
    for i in range(start, end):
        is_peak = signal[i] > signal[i - 1] and signal[i] > signal[i + 1]
        is_trough = signal[i] < signal[i - 1] and signal[i] < signal[i + 1]

        if mode == "peak" and is_peak:
            candidates.append(i)
        elif mode == "trough" and is_trough:
            candidates.append(i)

    if lower_bound is not None:
        candidates = [i for i in candidates if i > lower_bound]
    if upper_bound is not None:
        candidates = [i for i in candidates if i < upper_bound]

    if len(candidates) == 0:
        return pred_index
    return min(candidates, key=lambda i: abs(i - pred_index))


# %% 

data_dir = save_root/"data"
summary_dir = save_root/"summary"
log_dir = save_root/"logs"

for folder in [data_dir, summary_dir, log_dir]:
    folder.mkdir(parents=True, exist_ok=True)

run_log_path = log_dir/f"run_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"


def run_log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)

    try:
        with open(run_log_path, "a") as f:
            f.write(line + "\n")
    except OSError as error:
        print(f"cant write file: {error}")


def update_csv(df, final_path):
    final_path = Path(final_path)
    last_error = None

    for delay in [0, 0.5, 1, 2, 5]:
        temp_path = None

        try:
            if delay > 0:
                time.sleep(delay)

            with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=final_path.parent, suffix=".tmp") as f:
                temp_path = f.name
                df.to_csv(f, index=False)
            os.replace(temp_path, final_path)
            return

        except OSError as error:
            last_error = error
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    raise last_error


def run_complete_check(metadata_path, summary_path):
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return False

    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return False

    try:
        metadata_df = pd.read_csv(metadata_path)
        return (len(metadata_df) == 1 and metadata_df.loc[0, "status"] == "completed")
    except Exception:
        return False
# %%

with open(combined_data_path, "rb") as f:
    data = pickle.load(f)

signals_ids = data["signals_id"]    
abr_signals = data["X_data"]
peak_targets = data["y_data"]
reference_signals = data["reference_data"]

signal_length = len(abr_signals[0])
num_classes = signal_length
all_indices = list(range(len(abr_signals)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("n abrs:", len(abr_signals))
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %% 

remove_fracs = [x / 100 for x in range(0, 100, 5)] + [0.975]
repeats = range(1, 11)

planned_runs = []
for remove_frac in remove_fracs:
    for repeat_id in repeats:
        run_id = f"{strat}_rm{remove_frac:g}_r{repeat_id:02d}"
        planned_runs.append((run_id, remove_frac, repeat_id))

completed_existing = 0
for run_id, _, _ in planned_runs:
    data_path = data_dir/f"data_{run_id}.csv"
    summary_path = summary_dir / f"summary_{run_id}.csv"
    if run_complete_check(data_path, summary_path):
        completed_existing += 1

run_log(f"total planned runs: {len(planned_runs)}")
run_log(f"already complete: {completed_existing}")

attempted = 0
completed_now = 0
failed_now = 0
skipped = 0


# %%

for run_number, (run_id, remove_frac, repeat_id) in enumerate(planned_runs, start=1):
    data_path = data_dir/f"data_{run_id}.csv"
    summary_path = summary_dir/f"summary_{run_id}.csv"

    if run_complete_check(data_path, summary_path):
        skipped += 1
        run_log(f"[{run_number}/{len(planned_runs)}] skip complete: {run_id}")
        continue

    attempted += 1
    run_start_wall = time.time()
    run_start_perf = time.perf_counter()

    try:
        seed_generator = np.random.default_rng()
        sub_seed = int(seed_generator.integers(0, 1000001))
        cv_seed = int(seed_generator.integers(0, 1000001))
        train_seed = int(seed_generator.integers(0, 1000001))

        run_log(f"[{run_number}/{len(planned_runs)}] start: {run_id} - remove={remove_frac * 100}%")
        run_log(f"sub_seed={sub_seed}, cv_seed={cv_seed}, train_seed={train_seed}")

        subset_seed_generator = np.random.default_rng(sub_seed)
        indices_by_stim = {st: [] for st in stimuli}

        for idx in all_indices:
            stim = signals_ids[idx].split(":", 1)[0].lower()
            if stim in indices_by_stim:
                indices_by_stim[stim].append(idx)

        removed_indices = set()
        for stim in stimuli:
            stim_indices = indices_by_stim[stim]
            n_remove = round(remove_frac * len(stim_indices))

            if n_remove > 0:
                removed_here = subset_seed_generator.choice(stim_indices,size=n_remove, replace=False).tolist()
                removed_indices.update(removed_here)

        kept_indices = [idx for idx in all_indices if idx not in removed_indices]
        removed_indices = sorted(removed_indices)
        kept_indices = sorted(kept_indices)

        if len(kept_indices) < k:
            raise RuntimeError(f"only {len(kept_indices)} abrs kept for cv")

        run_log(f"subset: total={len(all_indices)}, kept={len(kept_indices)}, removed={len(removed_indices)}")

        abr_sub = [abr_signals[idx] for idx in kept_indices]
        ref_sub = [reference_signals[idx] for idx in kept_indices]
        target_sub = [peak_targets[idx] for idx in kept_indices]

        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_seed)

        indices = list(range(len(abr_sub)))
        
        fold_rng = random.Random(cv_seed)
        fold_rng.shuffle(indices)
        
        fold_size = len(indices) // k
        folds = []
        
        for i in range(k):
            if i == k - 1:
                fold = indices[i * fold_size:]
            else:
                fold = indices[i * fold_size:(i + 1) * fold_size]
            folds.append(fold)
        
        criterion = nn.CrossEntropyLoss()
        raw_errors = []
        adjusted_errors = []
        adjusted_correct = []
        
        raw_errors_per_target = {j: [] for j in range(num_targets)}
        adjusted_errors_per_target = {j: [] for j in range(num_targets)}
        adjusted_correct_per_target = {j: [] for j in range(num_targets)}


        for i in range(k):
            fold_number = i + 1
            run_log(f"{run_id}: fold {fold_number}/{k}")

            test_indices = folds[i]
            train_indices = []
            for j in range(k):
                if j != i:
                    train_indices += folds[j]

            X_train = [abr_sub[idx] for idx in train_indices]
            X_ref_train = [ref_sub[idx] for idx in train_indices]
            y_train = [target_sub[idx] for idx in train_indices]

            X_test = [abr_sub[idx] for idx in test_indices]
            X_ref_test = [ref_sub[idx] for idx in test_indices]
            y_test = [target_sub[idx] for idx in test_indices]

            train_dataset = peaksDataset(X_train, X_ref_train, y_train)
            test_dataset = peaksDataset(X_test, X_ref_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            model = peaksCNN(signal_length, hp, num_classes, num_targets).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
            
            for epoch in range(hp["epochs"]):
                train_loss = train_model_epoch(model, train_loader, optimizer, criterion, device, num_classes)

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    run_log(f"{run_id}: fold {fold_number}, epoch {epoch + 1}/{hp['epochs']}, training loss={train_loss:.6f}")
            model.eval()

            with torch.no_grad():
                for signals, targets in test_loader:
                    signals = signals.to(device)
                    targets = targets.to(device)

                    outputs = model(signals)
                    raw_pred = outputs.argmax(dim=2).squeeze(0).cpu().numpy()

                    prim_signal = signals.squeeze(0).cpu().numpy()[0]

                    adj2 = adjust_prediction(prim_signal, raw_pred[2], mode='peak', window=20, lower_bound=None, upper_bound=None)
                    adj3 = adjust_prediction(prim_signal, raw_pred[3], mode='trough', window=20, lower_bound=adj2, upper_bound=None)
                    adj0 = adjust_prediction(prim_signal, raw_pred[0], mode='peak', window=20, lower_bound=None, upper_bound=adj2)
                    adj1 = adjust_prediction(prim_signal, raw_pred[1], mode='trough', window=20, lower_bound=adj0, upper_bound=adj2)
                    adjusted = np.array([adj0, adj1, adj2, adj3])

                    target_idx = targets.squeeze(0).cpu().numpy()
                    
                    for j in range(num_targets):   
                        true_idx = target_idx[j]
                    
                        raw_error = abs(raw_pred[j] - true_idx)
                        adjusted_error = abs(adjusted[j] - true_idx)
                        is_corr = adjusted_error == 0
                    
                        raw_errors.append(raw_error)
                        adjusted_errors.append(adjusted_error)
                        adjusted_correct.append(is_corr)
                    
                        raw_errors_per_target[j].append(raw_error)
                        adjusted_errors_per_target[j].append(adjusted_error)
                        adjusted_correct_per_target[j].append(is_corr)

            del model, optimizer, train_loader, test_loader, train_dataset, test_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        initial_mae_ms = float(np.mean(raw_errors) * time_per_index)
        adjusted_mae_ms = float(np.mean(adjusted_errors) * time_per_index)
        adjusted_accuracy_pct = float(np.mean(adjusted_correct) * 100)
        runtime_sec = time.perf_counter() - run_start_perf
 
        
        summary_rows = []
        
        summary_rows.append({"run_id": run_id,
                             "strat": strat,
                             "remove_frac": remove_frac,
                             "keep_frac": 1 - remove_frac,
                             "repeat_id": repeat_id,
                             "target": "all targets",
                             "initial_mae_ms": initial_mae_ms,
                             "adjusted_mae_ms": adjusted_mae_ms,
                             "adjusted_accuracy_pct": adjusted_accuracy_pct})
        
        for j in range(num_targets):
            summary_rows.append({"run_id": run_id,
                                 "strat": strat,
                                 "remove_frac": remove_frac,
                                 "keep_frac": 1 - remove_frac,
                                 "repeat_id": repeat_id,
                                 "target": f"target{j + 1}",
                                 "initial_mae_ms": float(np.mean(raw_errors_per_target[j]) * time_per_index),
                                 "adjusted_mae_ms": float(np.mean(adjusted_errors_per_target[j]) * time_per_index),
                                 "adjusted_accuracy_pct": float(np.mean(adjusted_correct_per_target[j]) * 100)})
        
        summary_df = pd.DataFrame(summary_rows)

        data_df = pd.DataFrame([{"run_id": run_id,
                                 "status": "completed",
                                 "strat": strat,
                                 "remove_frac": remove_frac,
                                 "repeat_id": repeat_id,
                                 "sub_seed": sub_seed,
                                 "cv_seed": cv_seed,
                                 "train_seed": train_seed,
                                 "n_total": len(all_indices),
                                 "n_kept": len(kept_indices),
                                 "n_removed": len(removed_indices),
                                 "runtime_sec": runtime_sec}])

        update_csv(summary_df, summary_path)
        update_csv(data_df, data_path)

        completed_now += 1
        run_log(f"[{run_number}/{len(planned_runs)}] completed: {run_id} - initial MAE={initial_mae_ms:.4f} ms, adjusted MAE={adjusted_mae_ms:.4f} ms, adjusted accuracy={adjusted_accuracy_pct:.2f}%")
            
        
    except Exception as error:
        failed_now += 1
        traceback_text = traceback.format_exc()
    
        run_log(f"[{run_number}/{len(planned_runs)}] failed: {run_id} - {repr(error)}")
        run_log(traceback_text)
    
        failed_df = pd.DataFrame([{"run_id": run_id,
                                   "status": "failed",
                                   "strat": strat,
                                   "remove_frac": remove_frac,
                                   "repeat_id": repeat_id,
                                   "error": repr(error)}])
        try:
            update_csv(failed_df, data_path)
        except Exception as save_error:
            run_log(f"cant save failed data: {save_error}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()





