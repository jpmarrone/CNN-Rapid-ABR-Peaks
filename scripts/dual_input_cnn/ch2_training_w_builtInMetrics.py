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
outputs_folder = os.path.join(base_folder, "outputs", "dualChannel_newRuns")

batch_id = time.strftime("%Y%m%d_%H%M%S")
save_folder = os.path.join(outputs_folder, f"ch2_cv_check_{batch_id}")

os.makedirs(save_folder, exist_ok=True)

training_data_path = os.path.join(training_data_folder, "ch2_cnn_trainingData.pkl")

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
reference_signals = data["reference_data"]

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


#creates dataset input
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

        self.conv = nn.Sequential(self.conv_block(2, hp["conv1_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv1_channels"], hp["conv2_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv2_channels"], hp["conv3_channels"], kernel_size, padding, hp, slope),
                                  self.conv_block(hp["conv3_channels"], hp["conv4_channels"], kernel_size, padding, hp, slope))

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

#starts a timer before cv
cv_start = time.perf_counter()

dataset = peaksDataset(abr_signals, reference_signals, peak_targets)
num_samples = len(dataset)
print("dataset size:", num_samples)


#randomly shuffles the data indicies
seed = 12
random.seed(seed)

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
    X_ref_train = [reference_signals[idx] for idx in train_indices]
    y_train = [peak_targets[idx] for idx in train_indices]
    
    X_test = [abr_signals[idx] for idx in test_indices]
    X_ref_test=[reference_signals[idx] for idx in test_indices]
    y_test = [peak_targets[idx] for idx in test_indices]
    
    test_signal_ids = [signals_ids[idx] for idx in test_indices]
    
    train_dataset = peaksDataset(X_train, X_ref_train, y_train)
    test_dataset = peaksDataset(X_test, X_ref_test, y_test)
    
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
            ref_signal = arr[1]
            
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
                
                
                #for exporting the stats later
                trial_rows.append({"signal_id": signal_id, "subject": rat, "group": grp, "frequency": st,
                                   "day": get_day(signal_id), "sound_level": lvl, "target": f"target{j + 1}", "is_corr": is_corr})
                                   
            wrong = np.where(adjusted != target_idx)[0].tolist()
            if wrong:
                plt.figure(figsize=(6, 4.5))

                plt.plot(t, ref_signal, label='Channel 1', linewidth=1.5, color='gray')
                plt.plot(t, prim_signal, label='Channel 2', linewidth=2.5)
                
                pred_x = [t[adjusted[j]] for j in range(num_targets)]
                pred_y = [prim_signal[adjusted[j]] for j in range(num_targets)]
            
                targ_x = [t[target_idx[j]] for j in range(num_targets)]
                targ_y = [prim_signal[target_idx[j]] for j in range(num_targets)]
            
                plt.scatter(targ_x, targ_y, marker='X', color='red', s=48, label="Ground Truth", zorder=2)
                plt.scatter(pred_x, pred_y, marker='o', color='black', s=55, label="Prediction", zorder=3)
                
                plt.xlabel("Time (ms)")
                plt.ylabel("ABR Amplitude (Normalized)")
                plt.title(f"{signal_id} - wrong targets: {wrong}")
                plt.ylim(-0.05, 1.05)
                plt.xlim(-0.5, 10.5)
                plt.legend(fontsize=12)
                
                ax = plt.gca()
                ax.spines['top'].set_visible(False) 
                ax.spines['right'].set_visible(False)

                plt.tight_layout()
                plt.show()

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


#%% accuracy plotting metrics

#overall accuracy per sound level plot
acc_overall = []
for sl in plot_levels:
    total = global_level_total[sl]
    correct = global_level_correct[sl]
    acc_overall.append(((correct / total) * 100) if total > 0 else np.nan)

plt.figure(figsize=(6, 4.5))
plt.plot(plot_levels, acc_overall, marker='o', linewidth=2.5, markersize=11, label="Channel 2")
plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Overall", weight='bold')
plt.xticks(plot_levels)
plt.ylim(67, 103)
plt.grid(False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_yticks(range(85, 101, 5))
# ax.set_yticklabels([str(y) for y in range(85, 101, 5)])
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()


#overall accuracy per sound level per target
overall_acc = []
for j in range(num_targets):
    target_acc = []
    for sl in plot_levels:
        total = global_level_total_per_target[j][sl]
        correct = global_level_correct_per_target[j][sl]
        if total > 0:
            target_acc.append((correct / total) * 100)
        else:
            target_acc.append(np.nan)
    overall_acc.append(target_acc)
    
plt.figure(figsize=(6, 4.5))
for j in range(num_targets):
    ys = overall_acc[j]
    plt.plot(plot_levels, ys, marker='o', label=label_names[j], markersize=11, linewidth=2.5)

plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Channel 2 - Overall", weight='bold')
plt.xticks(plot_levels)
plt.ylim(67, 103)
plt.grid(False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_yticks(range(85, 101, 5))
# ax.set_yticklabels([str(y) for y in range(85, 101, 5)])
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()


#per stimulus accuracy per sound level
plt.figure(figsize=(6, 4.5))
for st in stimuli:
    ys = []
    for sl in plot_levels:
        total = global_stim_level_total[st][sl]
        correct = global_stim_level_correct[st][sl]
        if total > 0:
            ys.append((correct / total) * 100)
        else:
            ys.append(np.nan) 
    plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=stim_labels.get(st, st))

plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Channel 2 - Ind. Stimuli", weight='bold')
plt.xticks(plot_levels)
# plt.ylim(89, 101)
plt.grid(False)
plt.legend(fontsize=12, loc="lower right")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


#per target accuaracy per stimulus per soundlevel
for j in range(num_targets):
    plt.figure(figsize=(6, 4.5))

    for st in stimuli:
        ys = []
        for sl in plot_levels:
            total = global_stim_level_total_per_target[j][st][sl]
            correct = global_stim_level_correct_per_target[j][st][sl]
            if total > 0:
                ys.append((correct / total) * 100)
            else:
                ys.append(np.nan)
        plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=stim_labels.get(st, st))

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Channel 2 - {label_names[j]}", weight='bold')
    plt.xticks(plot_levels)
    plt.ylim(64, 106)
    plt.grid(False)
    plt.legend(fontsize=12, loc="lower right")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_yticks(range(65, 101, 5))
    # ax.set_yticklabels([str(y) for y in range(65, 101, 5)])
    plt.tight_layout()
    plt.show()


#per stimulus accuracy per target per sound level
for st in stimuli:
    plt.figure(figsize=(6, 4.5))

    for j in range(num_targets):
        ys = []
        for sl in plot_levels:
            total = global_stim_level_total_per_target[j][st][sl]
            correct = global_stim_level_correct_per_target[j][st][sl]
            if total > 0:
                ys.append((correct / total) * 100)
            else:
                ys.append(np.nan)
        plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=label_names[j])

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Channel 2 - {stim_labels.get(st, st)}", weight="bold")
    plt.xticks(plot_levels)
    # plt.ylim(89, 101)
    plt.grid(False)
    plt.legend(fontsize=12, loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    
#%% movement plots

#overall movement plot
avg_move_ms_overall = []
for sl in plot_levels:
    vals = []
    for j in range(num_targets):
        vals += movement_ms_per_target[j][sl]
    if len(vals) > 0:
        avg_move_ms_overall.append(np.mean(vals))
    else:
        avg_move_ms_overall.append(np.nan)

plt.figure(figsize=(6, 4.5))
plt.plot(plot_levels, avg_move_ms_overall, marker='o', linewidth=2.5, markersize=11)

plt.xlabel("Sound Level (dB)")
plt.ylabel("Prediction Adjustment (ms)")
plt.title("Channel 2 - Overall", weight="bold")
plt.xticks(plot_levels)
# plt.ylim(-0.01, 0.11)
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


#movement per target
avg_movement_time = []
for j in range(num_targets):
    target_move = []
    
    for sl in plot_levels:
        vals = movement_ms_per_target[j][sl]
        if len(vals) > 0:
            target_move.append(np.mean(vals))
        else:
            target_move.append(np.nan)
    avg_movement_time.append(target_move)

plt.figure(figsize=(6, 4.5))
for j in range(num_targets):
    ys = avg_movement_time[j]
    plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=label_names[j])
    
plt.xlabel("Sound Level (dB)")
plt.ylabel("Prediction Adjustment (ms)")
plt.title("Channel 2 - Overall", weight="bold")
plt.xticks(plot_levels)
plt.grid(False)
plt.legend(fontsize=12, loc="upper right")
plt.ylim(-0.01, 0.16)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([0.00, 0.05, 0.10, 0.15])
ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15'])
plt.tight_layout()  
plt.show()


#movement per stimulus 
avg_move_ms_by_stim = []
for st in stimuli:
    stim_move = []
    for sl in plot_levels:
        vals = []
        for j in range(num_targets):
            vals += movement_ms_per_target_stim[j][st][sl]
        if len(vals) > 0:
            stim_move.append(np.mean(vals))
        else:
            stim_move.append(np.nan)
    avg_move_ms_by_stim.append(stim_move)

plt.figure(figsize=(6, 4.5))
for i in range(len(stimuli)):
    st = stimuli[i]
    ys = avg_move_ms_by_stim[i]
    plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=stim_labels.get(st, st))
    
plt.xlabel("Sound Level (dB)")
plt.ylabel("Prediction Adjustment (ms)")
plt.title("Channel 2 - Ind. Stimuli", weight="bold")
plt.xticks(plot_levels)
plt.grid(False)
plt.legend(fontsize=12, loc="upper right")
plt.ylim(-0.01, 0.16)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([0.00, 0.05, 0.10, 0.15])
ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15'])
plt.tight_layout()
plt.show()


#movement per stimulus per target
for st in stimuli:
    plt.figure(figsize=(6, 4.5))
    for j in range(num_targets):
        ys = []
        for sl in plot_levels:
            vals = movement_ms_per_target_stim[j][st][sl]
            if len(vals) > 0:
                ys.append(np.mean(vals))
            else:
                ys.append(np.nan)
        plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=label_names[j])

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Prediction Adjustment (ms)")
    plt.title(f"Channel 2 - {stim_labels.get(st, st)}", weight="bold")
    plt.xticks(plot_levels)
    plt.grid(False)
    plt.legend(fontsize=12, loc="upper right")
    # plt.ylim(-0.01, 0.16)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15'])
    plt.tight_layout()
    plt.show()


#movement per target per stimulus
for j in range(num_targets):
    plt.figure(figsize=(6, 4.5))
    for st in stimuli:
        ys = []
        for sl in plot_levels:
            vals = movement_ms_per_target_stim[j][st][sl]
            if len(vals) > 0:
                ys.append(np.mean(vals))
            else:
                ys.append(np.nan)
        plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=stim_labels.get(st, st))
        
    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Prediction Adjustment (ms)")
    plt.title(f"Channel 2 - {label_names[j]}", weight="bold")
    plt.xticks(plot_levels)
    plt.grid(False)
    plt.legend(fontsize=12, loc="upper right")
    # plt.ylim(-0.01, 0.16)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15'])
    plt.tight_layout()
    plt.show()
    
    
    
#%% exposure grouping comparison plots

#saf exposure
plt.figure(figsize=(6, 4.5))
for g in exp_groups:
    ys = []
    for sl in plot_levels:
        total = global_exp_level_total[g][sl]
        correct = global_exp_level_correct[g][sl]

        if total > 0:
            ys.append((correct / total) * 100)
        else:
            ys.append(np.nan)
    plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=g)

plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Channel 2 - Overall", weight="bold")
plt.xticks(plot_levels)
plt.ylim(89, 101)
plt.grid(False)
plt.legend(fontsize=12, loc="lower right")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


#exposure group per stim
for st in stimuli:
    plt.figure(figsize=(6, 4.5))

    for g in exp_groups:
        ys = []
        for sl in plot_levels:
            total = global_exp_stim_level_total[g][st][sl]
            correct = global_exp_stim_level_correct[g][st][sl]
            if total > 0:
                ys.append((correct / total) * 100)
            else:
                ys.append(np.nan)
        plt.plot(plot_levels, ys, marker='o', linewidth=2.5, markersize=11, label=g)

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Channel 2 - {stim_labels.get(st, st)}", weight="bold")
    plt.xticks(plot_levels)
    # plt.ylim(84, 101)
    plt.grid(False)
    plt.legend(fontsize=12, loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks(range(85, 101, 5))
    ax.set_yticklabels([str(y) for y in range(85, 101, 5)])
    plt.tight_layout()
    plt.show()
    
    

#%% probability plots

#overall by target
for j, tname in enumerate(label_names):
    plt.figure(figsize=(6, 4.5))

    plt.axhline(0.9, color="gray", linestyle="--", linewidth=1.5,label="90% probability")
    for sl in sorted(plot_levels, reverse=True):
        data_vals = movement_ms_per_target[j][sl]
        if len(data_vals) > 0:
            res = stats.ecdf(data_vals)
            res.cdf.plot(plt.gca(), label=f"{sl} dB", linewidth=2.5)

    plt.xlabel("Prediction Adjustment (ms)")
    plt.ylabel("Probability")
    plt.title(f"Channel 2 - {tname}", weight="bold")
    plt.xlim(-0.05, 0.85)
    plt.ylim(-0.02, 1.02)
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    plt.show()
    
    
    
#all targets per stimulus
for st in stimuli:
    st_name = stim_labels.get(st, st)
    plt.figure(figsize=(6, 4.5))

    plt.axhline(0.9, color="gray", linestyle="--", linewidth=1.5, label="90% probability")

    for sl in sorted(plot_levels, reverse=True):
        data_vals = []
        for j in range(num_targets):
            data_vals += movement_ms_per_target_stim[j][st][sl]
        if len(data_vals) > 0:
            res = stats.ecdf(data_vals)
            res.cdf.plot(plt.gca(), label=f"{sl} dB", linewidth=2.5)

    plt.xlabel("Prediction Adjustment (ms)")
    plt.ylabel("Probability")
    plt.title(f"Channel 2 - {st_name}", weight="bold")
    plt.xlim(-0.05, 0.85)
    plt.ylim(-0.02, 1.02)
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    plt.show()  
  
    
    
#per stimulus and target
for st in stimuli:
    st_name = stim_labels.get(st, st)
    for j, tname in enumerate(label_names):
        plt.figure(figsize=(6, 4.5))

        plt.axhline(0.9, color="gray", linestyle="--", linewidth=1.5, label="90% probability")

        for sl in sorted(plot_levels, reverse=True):
            data_vals = movement_ms_per_target_stim[j][st][sl]
            if len(data_vals) > 0:
                res = stats.ecdf(data_vals)
                res.cdf.plot(plt.gca(), label=f"{sl} dB", linewidth=2.5)

        plt.xlabel("Prediction Adjustment (ms)")
        plt.ylabel("Probability")
        plt.title(f"Channel 2 - {st_name} - {tname}", weight="bold")
        plt.xlim(-0.05, 0.85)
        plt.ylim(-0.02, 1.02)
        plt.grid(False)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(fontsize=12, loc="lower right")
        plt.tight_layout()
        plt.show()
        
   
        
#%% calibration curve plotting

n_bins = 5

#all stim per target
plt.figure(figsize=(6, 4.5))

for j in range(num_targets):
    y_prob = np.asarray(init_conf_per_target[j])
    y_true = np.asarray(init_corr_per_target[j])

    if len(y_prob) > 0:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        plt.plot(prob_pred, (prob_true * 100), marker="o", label=label_names[j], markersize=11, linewidth=2.5)
plt.plot([0, 1], [0, 100], linestyle="--", color="gray", linewidth=1.5)

plt.title("Channel 2 - Overall", weight="bold")
plt.xlabel("Softmax Probability")
plt.ylabel("Accuracy (%)")
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
# ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
# ax.set_yticks(range(0, 101, 25))
# ax.set_yticklabels([str(y) for y in range(0, 101, 25)])
plt.legend(fontsize=12, loc="lower right")
plt.tight_layout()
plt.show()  



#all targets per stim
plt.figure(figsize=(6, 4.5))
for st in stimuli:
    y_prob = np.asarray(init_conf_per_stim[st])
    y_true = np.asarray(init_corr_per_stim[st])

    if len(y_prob) > 0:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        plt.plot(prob_pred, (prob_true * 100), marker="o", label=stim_labels.get(st, st), markersize=11, linewidth=2.5)
plt.plot([0, 1], [0, 100], linestyle="--", color="gray", linewidth=1.5)

plt.title("Channel 2 - Ind. Stimuli", weight="bold")
plt.xlabel("Softmax Probability")
plt.ylabel("Accuracy (%)")
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.legend(fontsize=12, loc="lower right")
plt.tight_layout()
plt.show()


#per stimulus split by target
for st in stimuli:
    plt.figure(figsize=(6, 4.5))

    for j in range(num_targets):
        y_prob = np.asarray(init_conf_per_target_stim[st][j])
        y_true = np.asarray(init_corr_per_target_stim[st][j])

        if len(y_prob) > 0:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
            plt.plot(prob_pred, (prob_true * 100), marker="o", label=label_names[j], markersize=11, linewidth=2.5)
    plt.plot([0, 1], [0, 100], linestyle="--", color="gray", linewidth=1.5)

    plt.title(f"Channel 2 - {stim_labels.get(st, st)}", weight="bold")
    plt.xlabel("Softmax Probability")
    plt.ylabel("Accuracy (%)")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    plt.show()



#%% quantification stuff

#movement distance quantified
rows_shift = []
for j, tname in enumerate(label_names):
    vals = []
    for sl in plot_levels:
        vals += movement_ms_per_target[j][sl]

    n, mean, median, std = sum_vals(vals)
    rows_shift.append({"stimulus": "Overall", "target": tname, "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})

#all targets
vals_all = []
for j in range(num_targets):
    for sl in plot_levels:
        vals_all += movement_ms_per_target[j][sl]

n, mean, median, std = sum_vals(vals_all)
rows_shift.append({"stimulus": "Overall", "target": "all targets", "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})


for st in stimuli:
    vals = []
    for j in range(num_targets):
        for sl in plot_levels:
            vals += movement_ms_per_target_stim[j][st][sl]

    n, mean, median, std = sum_vals(vals)
    st_name = stim_labels.get(st, st)
    rows_shift.append({"stimulus": st_name, "target": "all targets", "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})    


for st in stimuli:
    st_name = stim_labels.get(st, st)
    for j, tname in enumerate(label_names):
        vals = []
        for sl in plot_levels:
            vals += movement_ms_per_target_stim[j][st][sl]

        n, mean, median, std = sum_vals(vals)
        rows_shift.append({"stimulus": st_name, "target": tname, "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})

df_shift_stats = pd.DataFrame(rows_shift)
stim_order = ["Overall"] + [stim_labels.get(st, st) for st in stimuli]
target_order = label_names + ["all targets"]

df_shift_stats["stimulus"] = pd.Categorical(df_shift_stats["stimulus"], categories=stim_order, ordered=True)
df_shift_stats["target"] = pd.Categorical(df_shift_stats["target"], categories=target_order, ordered=True)
df_shift_stats = df_shift_stats.sort_values(["stimulus", "target"]).reset_index(drop=True)

print("\ntime shift stats (ms):")
print(df_shift_stats.round(4).to_string(index=False))



#%% movement distance probability

rows_p90_time = []

for j, tname in enumerate(label_names):
    for sl in plot_levels:
        vals_time = movement_ms_per_target[j][sl]
        if len(vals_time) > 0:
            p90_ms = np.quantile(vals_time, 0.9)
        else:
            p90_ms = np.nan
            
        rows_p90_time.append({"stimulus": "Overall", "target": tname, "sound_level": sl, "p90_ms": p90_ms})


for sl in plot_levels:
    vals_time = []
    for j in range(num_targets):
        vals_time += movement_ms_per_target[j][sl]
    if len(vals_time) > 0:
        p90_ms = np.quantile(vals_time, 0.9)
    else:
        p90_ms = np.nan

    rows_p90_time.append({"stimulus": "Overall", "target": "all targets", "sound_level": sl, "p90_ms": p90_ms})


for st in stimuli:
    st_name = stim_labels.get(st, st)
    for j, tname in enumerate(label_names):
        for sl in plot_levels:
            vals_time = movement_ms_per_target_stim[j][st][sl]
            if len(vals_time) > 0:
                p90_ms = np.quantile(vals_time, 0.9)
            else:
                p90_ms = np.nan

            rows_p90_time.append({"stimulus": st_name, "target": tname, "sound_level": sl, "p90_ms": p90_ms})


for st in stimuli:
    st_name = stim_labels.get(st, st)
    for sl in plot_levels:
        vals_time = []
        for j in range(num_targets):
            vals_time += movement_ms_per_target_stim[j][st][sl]
        if len(vals_time) > 0:
            p90_ms = np.quantile(vals_time, 0.9)
        else:
            p90_ms = np.nan

        rows_p90_time.append({"stimulus": st_name, "target": "all targets", "sound_level": sl, "p90_ms": p90_ms})

df_p90_time = pd.DataFrame(rows_p90_time)
df_p90_time = df_p90_time.pivot_table(index=["stimulus", "target"], columns="sound_level", values="p90_ms").reset_index()

df_p90_time["stimulus"] = pd.Categorical(df_p90_time["stimulus"], categories=stim_order, ordered=True)
df_p90_time["target"] = pd.Categorical(df_p90_time["target"], categories=target_order, ordered=True)
df_p90_time = df_p90_time.sort_values(["stimulus", "target"]).reset_index(drop=True)

print("\n90th percentile adjustment distance (ms):")
print(df_p90_time.round(4).to_string(index=False))




#%% 

#incorrect predictions disance from ground truth....after adjustments
rows_err = []

#overall per target
for j, tname in enumerate(label_names):
    vals = []
    for sl in plot_levels:
        vals += err_ms_per_target[j][sl]

    n, mean, median, std = sum_vals(vals)
    rows_err.append({"stimulus": "Overall","target": tname, "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})


vals_all = []
for j in range(num_targets):
    for sl in plot_levels:
        vals_all += err_ms_per_target[j][sl]

n, mean, median, std = sum_vals(vals_all)
rows_err.append({"stimulus": "Overall", "target": "all targets", "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})



for st in stimuli:
    vals = []
    for j in range(num_targets):
        for sl in plot_levels:
            vals += err_ms_per_target_stim[st][j][sl]

    n, mean, median, std = sum_vals(vals)
    st_name = stim_labels.get(st, st)
    rows_err.append({"stimulus": st_name, "target": "all targets", "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})


for st in stimuli:
    st_name = stim_labels.get(st, st)
    for j, tname in enumerate(label_names):
        vals = []
        for sl in plot_levels:
            vals += err_ms_per_target_stim[st][j][sl]

        n, mean, median, std = sum_vals(vals)
        rows_err.append({"stimulus": st_name, "target": tname, "n": n, "mean_ms": mean, "median_ms": median, "std_ms": std})

df_err_stats = pd.DataFrame(rows_err)
df_err_stats["stimulus"] = pd.Categorical(df_err_stats["stimulus"], categories=stim_order, ordered=True)
df_err_stats["target"] = pd.Categorical(df_err_stats["target"], categories=target_order, ordered=True)
df_err_stats = df_err_stats.sort_values(["stimulus", "target"]).reset_index(drop=True)

print("\nincorrect only distance from ground truth")
print(df_err_stats.round(4).to_string(index=False))



#%% save the trained model 
# #for actual application stuff...
    
# full_dataset = peaksDataset(abr_signals, reference_signals, peak_targets)
# full_loader = DataLoader(full_dataset, batch_size=hp["batch_size"], shuffle=True)
# final_model = peaksCNN(signal_length, hp, num_classes, num_targets).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(final_model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])

# for epoch in range(hp["epochs"]):
#     train_loss = train_model_epoch(final_model, full_loader, optimizer, criterion, device, num_classes)

#     if epoch == 0 or (epoch + 1) % 10 == 0:
#         print(f"Final Model, Epoch {epoch + 1}/{hp['epochs']}, Training Loss: {train_loss:.6f}")

# checkpoint = {"model_state_dict": final_model.state_dict(), "hp": hp, "signal_length": signal_length,
#               "num_classes": num_classes, "num_targets": num_targets}

# torch.save(checkpoint, model_output_path)

