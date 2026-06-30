#%% import stuff

import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cycler import cycler

#%% plotting stuff

matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',  '#4DBEEE', '#A2142F', '#008080']
plt.rc('axes', prop_cycle=cycler(color=matlab_colors))
plt.rcParams.update({"font.family": "Arial", "font.size": 16})

# %% define filepaths

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(os.path.dirname(script_folder))

training_data_folder = os.path.join(base_folder, "data", "training")
model_folder = os.path.join(base_folder, "models")
outputs_folder = os.path.join(base_folder, "outputs")

os.makedirs(training_data_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(outputs_folder, exist_ok=True)

training_data_path = os.path.join(training_data_folder, "ch2_cnn_trainingData.pkl")
model_output_path = os.path.join(model_folder, "final_ch2_cnnSingleChannel.pth")

batch_id = time.strftime("%Y%m%d_%H%M%S")
run_folder = os.path.join(outputs_folder, f"ch2_10runs_{batch_id}")
os.makedirs(run_folder, exist_ok=True)

all_trials_path = os.path.join(run_folder, "ch2_all_trial_stats.csv")

with open(training_data_path, "rb") as f:
    data = pickle.load(f)



# %%
torch.cuda.empty_cache()

#hyperparameter definitions
hp = {'num_conv_layers': 4,
      'conv1_channels': 53,
      'conv2_channels': 90,
      'conv3_channels': 65,
      'conv4_channels': 65,
      'kernel_size': 3,
      'stride': 1,
      'use_batch_norm': True,
      'slope': 0.01,
      'pooling_type': 'max',
      'pooling_kernel_size': 2,
      'pooling_stride': 2,
      'dropout_conv': 0.08755577559324501,
      'dropout_fc': 0.85,
      'num_fc_layers': 2,
      'fc_neurons': 414,
      'learning_rate': 0.002278056544827262,
      'weight_decay': 0.02,
      'batch_size': 100,
      'epochs': 250}

signals_ids = data["signals_id"]    
abr_signals = data["X_data"]
peak_targets = data["y_data"]

signal_length = abr_signals.shape[1]
num_classes = signal_length
#targets = wave peaks/troughs being predicted...4 for ch1 and 6 for ch2
num_targets = 6 

stim_labels = {"click": "Click",
               "4k": "4 kHz",
               "8k": "8 kHz",
               "10k": "10 kHz",
               "16k": "16 kHz"}
stimuli = list(stim_labels)

nruns = 10
seed_generator = np.random.default_rng()
channel = "Ch2"


# %% functions and definitions

#exposure group animals
functional = ["YM4n", "YM5al", "YM5n", "YM4al", "YM5an", "YM5ap", "YM4as", "YM5ae", "YM5as", "YM4g", "YM4m", "YM5ah",
              "YM5am", "YM5m", "YM5z", "YM4am", "YM4ao", "YM4ar", "YM5aa", "YM5ar", "YM5s"]

sham = ["YM4af", "YM4ag", "YM4at", "YM4au", "YM4av", "YM4aw", "YM4k", "YM4l", "YM4u", "YM5af", "YM5ag", "YM5at", "YM5au",
        "YM5av", "YM5aw", "YM5k", "YM5l", "YM5t", "YM5v"]
    

def get_sound_level(signal_id):
    return int(signal_id.split("_")[-1])

def get_ratname(signal_id):
    rest = signal_id.split(":")[1]
    return rest.split("_")[0]

def get_day(signal_id):
    rest = signal_id.split(":")[1]
    return rest.split("_")[1]

def get_exposure_group(signal_id):
    rat = get_ratname(signal_id)

    if rat in functional:
        return "Functional"
    elif rat in sham:
        return "Sham"
    else:
        return None
    
def get_stim(signal_id):
    return signal_id.split(":")[0].lower()


#target adjustment function....moves initial predictions to local extrema
    #enforces the ch2 target order
        #because the targets should be w1 peak, w1 trough, w4 peak, w4 trough, w5 peak, w5 trough
    #order goes like:
        #W1 peak adjusted
        #W1 trough adjusted using the adjusted W1 peak as prev_extreme 
        #W4 peak adjusted using the adjusted W1 trough as prev_extreme
        #W4 trough adjusted using the adjusted W4 peak as prev_extreme
        #W5 peak adjusted using the adjusted W4 trough as prev_extreme
        #W5 trough adjusted using the adjusted W5 peak as prev_extreme
        
def adjust_prediction(signal, pred_index, mode, window, lower_bound):
    start = max(1, pred_index - window)
    end = min(len(signal) - 1, pred_index + window + 1)

    if pred_index > 0 and pred_index < len(signal) - 1:
        is_trough = (signal[pred_index] < signal[pred_index - 1] and signal[pred_index] < signal[pred_index + 1])

        if mode == "trough" and is_trough:
            return pred_index

    candidates = []

    for i in range(start, end):
        is_peak = signal[i] > signal[i - 1] and signal[i] > signal[i + 1]
        is_trough = signal[i] < signal[i - 1] and signal[i] < signal[i + 1]

        if mode == "peak" and is_peak:
            candidates.append(i)
        elif mode == "trough" and is_trough:
            candidates.append(i)
            
    candidates = [i for i in candidates if i > lower_bound]

    if len(candidates) == 0:
        return pred_index

    #peaks use largest extrema, troughs use closest extrema
    if mode == "peak":
        return max(candidates, key=lambda i: signal[i])
    return min(candidates, key=lambda i: abs(i - pred_index))


#for the quantification stuff at the end
def sum_vals(vals):
    n = len(vals)

    if n == 0:
        return 0, np.nan, np.nan, np.nan

    mean = np.mean(vals)
    median = np.median(vals)
    #sample std
    std = np.std(vals, ddof=1) if n > 1 else 0
    return n, mean, median, std


class peaksDataset(Dataset):
    def __init__(self, abr_prim, peaks):
        abr_prim = np.stack(abr_prim).astype(np.float32)
        peaks = np.asarray(peaks, dtype=np.int64)

        self.signals = torch.from_numpy(abr_prim).unsqueeze(1)
        self.targets = torch.from_numpy(peaks)

    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.signals[idx], self.targets[idx]
    

#create the cnn
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

        self.conv = nn.Sequential(self.conv_block(1, hp["conv1_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv1_channels"], hp["conv2_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv2_channels"], hp["conv3_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv3_channels"], hp["conv4_channels"], kernel_size, padding, hp, slope))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
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

#specify training mode setup...returns average loss per epoch
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

# #starts a timer before cv
# cv_start = time.perf_counter()

dataset = peaksDataset(abr_signals, peak_targets)
num_samples = len(dataset)
print("dataset size:", num_samples)


# #randomly shuffles the data indicies
# seed = 12
# random.seed(seed)

for run_number in range(1, nruns + 1):

    cv_start = time.perf_counter()
    cv_seed = int(seed_generator.integers(0, 1000001))
    train_seed = int(seed_generator.integers(0, 1000001))

    random.seed(cv_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
        torch.cuda.empty_cache()

    print(f"Run {run_number}/{nruns}")

    indices = list(range(num_samples))
    random.shuffle(indices)
    
    #cv definition
    k = 10
    fold_size = num_samples // k
    folds = []
    
    for i in range(k):
        if i == k - 1:
            fold = indices[i * fold_size:]
        else:
            fold = indices[i * fold_size: (i + 1) * fold_size]
        folds.append(fold)
    
    fold_results = []
    all_fold_losses = []
    
    #names used for plots and tables
    label_names = ["W1 Peak", "W1 Trough", "W4 Peak", "W4 Trough", "W5 Peak", "W5 Trough"]
    sound_levels = list(range(10, 81, 10))
    plot_levels = list(range(10, 81, 10))
    exp_groups = ["Functional", "Sham"]
    
    time_per_index = 0.04095997267759563
    t = np.linspace(0, 10.240, signal_length)
    
    #accuracy counters...overall, sound level, stimulus, and target specific
    global_correct = 0
    global_total = 0
         
    global_level_correct = {sl: 0 for sl in sound_levels}
    global_level_total = {sl: 0 for sl in sound_levels}
    
    global_stim_level_correct_per_target = {j: {st: {sl: 0 for sl in sound_levels} for st in stimuli} for j in range(num_targets)}
    global_stim_level_total_per_target = {j: {st: {sl: 0 for sl in sound_levels} for st in stimuli} for j in range(num_targets)}
    
    global_stim_correct = {st: 0 for st in stimuli}
    global_stim_total = {st: 0 for st in stimuli}
    
    global_stim_level_correct = {st: {sl: 0 for sl in sound_levels} for st in stimuli}
    global_stim_level_total = {st: {sl: 0 for sl in sound_levels} for st in stimuli}
    
    global_level_correct_per_target = {j: {sl: 0 for sl in sound_levels} for j in range(num_targets)}
    global_level_total_per_target = {j: {sl: 0 for sl in sound_levels} for j in range(num_targets)}
    
    #movement from raw prediction to adjusted prediction...only when adjusted prediction is correct
    movement_ms_per_target = {j: {sl: [] for sl in sound_levels} for j in range(num_targets)}
    movement_ms_per_target_stim = {j: {st: {sl: [] for sl in sound_levels} for st in stimuli} for j in range(num_targets)}
    
    #distance from the ground truth for things that are still wrong after target adjustment
    err_ms_per_target = {j: {sl: [] for sl in sound_levels} for j in range(num_targets)}
    err_ms_per_target_stim = {st: {j: {sl: [] for sl in sound_levels} for j in range(num_targets)} for st in stimuli}
    
    #calibration info
    init_conf_per_target = {j: [] for j in range(num_targets)}
    init_corr_per_target = {j: [] for j in range(num_targets)}
    
    init_conf_per_stim = {st: [] for st in stimuli}
    init_corr_per_stim = {st: [] for st in stimuli}
    
    init_conf_per_target_stim = {st: {j: [] for j in range(num_targets)} for st in stimuli}
    init_corr_per_target_stim = {st: {j: [] for j in range(num_targets)} for st in stimuli}
    
    #saf exposure group
    global_exp_level_correct = {g: {sl: 0 for sl in sound_levels} for g in exp_groups}
    global_exp_level_total = {g: {sl: 0 for sl in sound_levels} for g in exp_groups}
    
    global_exp_stim_level_correct = {g: {st: {sl: 0 for sl in sound_levels} for st in stimuli} for g in exp_groups}
    global_exp_stim_level_total = {g: {st: {sl: 0 for sl in sound_levels} for st in stimuli} for g in exp_groups}
    
    trial_rows = []
    cont_trial_rows = []
    
    #uses gpu if available otherwise defaults to cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #%% run
    #cross validation loop
    for i in range(k):
        fold_number = i + 1
       
        print(f"\nFold {fold_number}:")
        #current fold is used as testing set
        #every other fold is used as training set
        test_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices += folds[j]
        
        X_train = [abr_signals[idx] for idx in train_indices]
        y_train = [peak_targets[idx] for idx in train_indices]
        
        X_test = [abr_signals[idx] for idx in test_indices]
        y_test = [peak_targets[idx] for idx in test_indices]
        
        test_signal_ids = [signals_ids[idx] for idx in test_indices]
        
        train_dataset = peaksDataset(X_train, y_train)
        test_dataset = peaksDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
        #test samples are not shuffled...model weights arent updated during testing so order deosnt matter
            #makes it easier to match ids to signals
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        model = peaksCNN(signal_length, hp, num_classes, num_targets).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
        
        #training loop....for each epoch
        train_losses = []
        for epoch in range(hp['epochs']):
            train_loss = train_model_epoch(model, train_loader, optimizer, criterion, device, num_classes)
            train_losses.append(train_loss)
        
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"Fold {fold_number}, Epoch {epoch + 1}/{hp['epochs']}, Training loss: {train_loss:.6f}")
        all_fold_losses.append(train_losses)
            
        #starts the testing part..
        model.eval()
        total_test_loss = 0
        correct_targets = 0
        total_targets = 0
        num_batches = 0
    
        fold_level_correct = {sl: 0 for sl in sound_levels}
        fold_level_total = {sl: 0 for sl in sound_levels}
    
        #dont calculate gradients during testing
        with torch.no_grad():
            for test_number, (signals, targets) in enumerate(test_loader):
                signal_id = test_signal_ids[test_number]
                signals = signals.to(device)
                targets = targets.to(device)
            
                st = get_stim(signal_id)
                lvl = get_sound_level(signal_id)
                rat = get_ratname(signal_id)
                grp = get_exposure_group(signal_id)
                
                num_batches += 1
                outputs = model(signals)
                probs = torch.softmax(outputs, dim=2)
                
           
                loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
                total_test_loss += loss.item()
    
    
                raw_pred = outputs.argmax(dim=2).squeeze(0).cpu().numpy()
                arr = signals.squeeze(0).cpu().numpy()
                prim_signal = arr[0]
                # ref_signal = arr[1]
                
                #do the target adjustment step...
                    #with the enforced order stuff
                adj0 = adjust_prediction(prim_signal, raw_pred[0], mode="peak", window=20, lower_bound = 0)
                adj1 = adjust_prediction(prim_signal, raw_pred[1], mode="trough", window=20, lower_bound=adj0)
                adj2 = adjust_prediction(prim_signal, raw_pred[2], mode="peak", window=20, lower_bound=adj1)
                adj3 = adjust_prediction(prim_signal, raw_pred[3], mode="trough", window=20, lower_bound=adj2)
                adj4 = adjust_prediction(prim_signal, raw_pred[4], mode="peak", window=20, lower_bound=adj3)
                adj5 = adjust_prediction(prim_signal, raw_pred[5], mode="trough", window=20, lower_bound=adj4)
                adjusted = np.array([adj0, adj1, adj2, adj3, adj4, adj5])
    
                target_idx = targets.squeeze(0).cpu().numpy()
                
                for j in range(num_targets):
                    raw_idx = raw_pred[j]
                    raw_conf = probs[0, j, raw_idx].item()
                    raw_corr = raw_idx == target_idx[j]
                                    
                    init_conf_per_target[j].append(raw_conf)
                    init_corr_per_target[j].append(raw_corr)
                    
                    if st in stimuli:
                        init_conf_per_stim[st].append(raw_conf)
                        init_corr_per_stim[st].append(raw_corr)
                        init_conf_per_target_stim[st][j].append(raw_conf)
                        init_corr_per_target_stim[st][j].append(raw_corr)
                
                for j in range(num_targets):
                    true_idx = target_idx[j]
                    pre_err = abs(raw_pred[j] - true_idx)
                    post_err = abs(adjusted[j] - true_idx)
                    is_corr = adjusted[j] == true_idx
    
        
                    #post adjustment distance from ground truth...only from predictions still incorrect after adjustmens
                    if post_err > 0:
                        err_ms_per_target[j][lvl].append(post_err * time_per_index)
                        
                        if st in stimuli:
                            err_ms_per_target_stim[st][j][lvl].append(post_err * time_per_index)
                    
                    #prediction movement...only for correct predictions (after the adjustment)
                    if is_corr:
                        move_idx = abs(adjusted[j] - raw_pred[j])
                        move_ms = move_idx * time_per_index
                        
                        movement_ms_per_target[j][lvl].append(move_ms)
                        
                        if st in stimuli:
                            movement_ms_per_target_stim[j][st][lvl].append(move_ms)
                                    
                    total_targets += 1
                    if is_corr:
                        correct_targets += 1
                    
                    fold_level_total[lvl] += 1
                    if is_corr:
                        fold_level_correct[lvl] += 1
                    
                    global_level_total[lvl] += 1
                    global_level_total_per_target[j][lvl] += 1
                    
                    if is_corr:
                        global_level_correct[lvl] += 1
                        global_level_correct_per_target[j][lvl] += 1
                    
                    #stimulus specific accuracy
                    if st in stimuli:
                        global_stim_total[st] += 1
                        global_stim_level_total[st][lvl] += 1
                        global_stim_level_total_per_target[j][st][lvl] += 1
                        
                        if is_corr:
                            global_stim_correct[st] += 1
                            global_stim_level_correct[st][lvl] += 1
                            global_stim_level_correct_per_target[j][st][lvl] += 1
                    
                    if grp in exp_groups:
                        global_exp_level_total[grp][lvl] += 1
                        if is_corr:
                            global_exp_level_correct[grp][lvl] += 1
                        
                        if st in stimuli:
                            global_exp_stim_level_total[grp][st][lvl] += 1
                            if is_corr:
                                global_exp_stim_level_correct[grp][st][lvl] += 1
                    

                    raw_conf = probs[0, j, raw_pred[j]].item()
                    raw_is_corr = raw_pred[j] == true_idx
                    adjustment_ms = abs(adjusted[j] - raw_pred[j]) * time_per_index
                    
                    trial_rows.append({"signal_id": signal_id,
                                       "fold": fold_number,
                                       "subject": rat,
                                       "group": grp,
                                       "frequency": st,
                                       "day": get_day(signal_id),
                                       "sound_level": lvl,
                                       "target": f"target{j + 1}",
                                       
                                       "true_idx": int(true_idx),
                                       "raw_pred_idx": int(raw_pred[j]),
                                       "adjusted_pred_idx": int(adjusted[j]),
                                       
                                       "raw_conf": raw_conf,
                                       "raw_is_corr": raw_is_corr,
                                       "is_corr": is_corr,
                                       
                                       "raw_error_ms": pre_err * time_per_index,
                                       "adjusted_error_ms": post_err * time_per_index,
                                       "adjustment_ms": adjustment_ms})
    
        avg_test_loss = total_test_loss / num_batches if num_batches else np.nan
        percent_acc = (correct_targets / total_targets) * 100 if total_targets else np.nan
        print(f"Fold {fold_number} evaluation:")
        print(f"Average test loss: {avg_test_loss:.6f}")
        print(f"Accuracy: {correct_targets}/{total_targets} ({percent_acc:.2f}%)")
        fold_results.append((avg_test_loss, percent_acc))
    
        global_correct += correct_targets
        global_total += total_targets

    
    #after all the folds are done.....
    if device.type == "cuda":
        torch.cuda.synchronize()
    cv_end = time.perf_counter()  
    
    #calculate global metrics...
    avg_loss = np.nanmean([fr[0] for fr in fold_results])
    global_accuracy = (global_correct / global_total) * 100 if global_total else np.nan
    total_cv_time = cv_end - cv_start
    
    print("\nTraining results:")
    print(f"Average test loss across folds: {avg_loss:.6f}")
    print(f"Average accuracy across folds: {global_accuracy:.2f}%")
    
    minutes, seconds = divmod(total_cv_time, 60)
    print(f"runtime: {minutes}m {seconds:.2f}s")
    
    print("\nAverage accuracy by stimulus:")
    for st in stimuli:
        total = global_stim_total[st]
        correct = global_stim_correct[st]
        acc = (correct / total) * 100 if total > 0 else np.nan
        print(f"{stim_labels.get(st, st)}: {acc:.2f}%")
        
    
    df_trials = pd.DataFrame(trial_rows)
    
    df_trials.insert(0, "batch_id", batch_id)
    df_trials.insert(1, "run_number", run_number)
    df_trials.insert(2, "cv_seed", cv_seed)
    df_trials.insert(3, "train_seed", train_seed)
    df_trials.insert(4, "channel", channel)
    
    df_trials = df_trials.sort_values(["subject", "frequency", "day", "sound_level", "target"]).reset_index(drop=True)
    run_trials_path = os.path.join(run_folder, f"ch2_trial_stats_run_{run_number:02d}.csv")
    
    df_trials.to_csv(run_trials_path, index=False)
    df_trials.to_csv(all_trials_path, mode="a", header=not os.path.exists(all_trials_path), index=False)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
