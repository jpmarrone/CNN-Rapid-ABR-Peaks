#%% overview



#%% setup

import os
import random
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime


script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(os.path.dirname(script_folder))

raw_data_folder = os.path.join(base_folder, "data", "raw")
model_folder = os.path.join(base_folder, "models")

batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(base_folder, "outputs", f"ch2_singleChannel_cnn_amplitudes_{batch_id}")

os.makedirs(output_folder, exist_ok=True)

mat_struct_path = os.path.join(raw_data_folder, "abr_structures_singleChannel_ch1.mat")
cnn_model_path = os.path.join(model_folder, "final_ch2_cnn_singleChannel.pth")

amp_output_name = "saf_ch2_singleChannel_abr_amplitudes"

# %% 

#filtering info
fs = 97656 / 4
lowcut = 80
highcut = 1500
order = 4

premean_range = slice(75, 91)
noisefloor_range = slice(69, 90)

#the mat file has a structure for each stimulus
stim_info = {"click": {"mat_key": "SAF_clickABR_unfiltered", "trunc": slice(90, 341),"t": np.linspace(3.6863975409836067, 13.96735068306011, 251)},
             "4k": {"mat_key": "SAF_4kABR_unfiltered", "trunc": slice(110, 361), "t": np.linspace(4.5056115344, 14.7456377488, 251)},
             "8k": {"mat_key": "SAF_8kABR_unfiltered", "trunc": slice(102, 353), "t": np.linspace(4.1779306955, 14.4179569100, 251)},
             "10k": {"mat_key": "SAF_10kABR_unfiltered", "trunc": slice(101, 352), "t": np.linspace(4.1369705906, 14.3769968051, 251)},
             "16k": {"mat_key": "SAF_16kABR_unfiltered", "trunc": slice(98, 349),"t": np.linspace(4.0140902761, 14.2541164905, 251)}}

plot_names = ["W1 P", "W1 T", "W4 P", "W4 T", "W5 P", "W5 T"]
marker_colors = ["red", "black", "red", "black", "red", "black"]
label_shift = 0.2

plt.rcParams.update({"font.family": "Arial", "font.size": 16})

#%% matlab structure to python dictionary conversion

def loadmat(filename):
    data = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
    return check_keys(data)

def check_keys(d):
    for k in d:
        if isinstance(d[k], scipy.io.matlab.mat_struct):
            d[k] = todict(d[k])
    return d

def todict(obj):
    d = {}
    for field in obj._fieldnames:
        x = getattr(obj, field)
        if isinstance(x, scipy.io.matlab.mat_struct):
            d[field] = todict(x)
        else:
            d[field] = x
    return d

#%% cnn part

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
    
# %% functions

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

# def get_recordings(stim, d):
#     records = []

#     for day, day_data in d.items():
#         for level, level_data in day_data.items():
#             ch1 = level_data["Ch1"]
#             ch2 = level_data["Ch2"]

#             for rat, sig1 in ch1.items():
#                 records.append({"stim": stim, "rat": rat, "day": day, "level": level, "id": f"{stim}:{rat}_{day}_{level}",
#                                 "ch1": np.asarray(sig1, dtype=float).ravel(), "ch2": np.asarray(ch2[rat], dtype=float).ravel()})
#     return records



def get_recordings(stim, d):
    records = []

    for day, day_data in d.items():
        for level, level_data in day_data.items():
            for rat, sig in level_data.items():
                records.append({"stim": stim, "rat": rat, "day": day, "level": level, "id": f"{stim}:{rat}_{day}_{level}",
                                "signal": np.asarray(sig, dtype=float).ravel()})
    return records

#%% run the actual extraction

#have the option to manually check each prediction....
    #it will plot the signal and the predictions one at a time
    #and then take user input for the correct labels if the predictions are incorrect
     
manual_check = input("manual check every prediction? (y/n): ").strip().lower() == "y"

if manual_check:
    start_stim = input("start stimulus (click, 4k, 10k, etc) (enter to start at beginning): ").strip().lower()
    start_rat = input("start rat (enter to start at beginning): ").strip().lower()
    start_day = input("start day (enter to start at beginning): ").strip().lower()
    start_level = input("start sound level (enter to start at beginning): ").strip()
else:
    start_stim = ""
    start_rat = ""
    start_day = ""
    start_level = ""

start_found = not manual_check
mat = loadmat(mat_struct_path)

#loads the cnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(cnn_model_path, map_location=device)

hp = checkpoint["hp"]
signal_length = checkpoint["signal_length"]
num_classes = checkpoint["num_classes"]
num_targets = checkpoint["num_targets"]

model = peaksCNN(signal_length, hp, num_classes, num_targets).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

nyq = 0.5 * fs
b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="bandpass")

amplitude_rows = []
random_plots = []
stop_now = False

for stim, info in stim_info.items():
    key = info["mat_key"]
    trunc = info["trunc"]
    t_trunc = info["t"]
    
    recs = get_recordings(stim, mat[key])

    x = np.vstack([r["signal"] for r in recs])
    x_raw = filtfilt(b, a, x, axis=1)
    x_prem = x_raw - x_raw[:, premean_range].mean(axis=1, keepdims=True)
    
    #normalize
    x_min = x_raw.min(axis=1, keepdims=True)
    x_max = x_raw.max(axis=1, keepdims=True)
    x_cnn = ((x_raw - x_min) / (x_max - x_min))[:, trunc]

    x_plot = x_prem[:, trunc]

    print(stim, len(recs), "ABRs")
    for i, rec in enumerate(recs):
        if manual_check and not start_found:
            stim_ok = (start_stim == "" or rec["stim"].lower() == start_stim.lower())
            rat_ok = (start_rat == "" or rec["rat"].lower() == start_rat.lower())
            day_ok = (start_day == "" or rec["day"].lower() == start_day.lower())
            level_ok = (start_level == "" or rec["level"] == start_level)
    
            if stim_ok and rat_ok and day_ok and level_ok:
                start_found = True
            else:
                continue

        cnn_input = x_cnn[i]
        # cnn_input = torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).to(device)
        cnn_input = torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(cnn_input)
        
        raw_pred = out.argmax(dim=2).squeeze(0).cpu().numpy()
           
        adj0 = adjust_prediction(x_cnn[i], raw_pred[0], mode="peak", window=20, lower_bound = 0)
        adj1 = adjust_prediction(x_cnn[i], raw_pred[1], mode="trough", window=20, lower_bound=adj0)
        adj2 = adjust_prediction(x_cnn[i], raw_pred[2], mode="peak", window=20, lower_bound=adj1)
        adj3 = adjust_prediction(x_cnn[i], raw_pred[3], mode="trough", window=20, lower_bound=adj2)
        adj4 = adjust_prediction(x_cnn[i], raw_pred[4], mode="peak", window=20, lower_bound=adj3)
        adj5 = adjust_prediction(x_cnn[i], raw_pred[5], mode="trough", window=20, lower_bound=adj4)
    
        final = [adj0, adj1, adj2, adj3, adj4, adj5]


        #if the manual check is chosen.....
            #is plots the premeaned truncated signal with the predcitons labeled
            #takes user input for if predictions are correct or not
            #then if incorrect...takes user input for the correct labels
        if manual_check:
            plt.figure(figsize=(7, 4))
            plt.plot(t_trunc, x_plot[i], label = "Channel 2", linewidth=2)

            for k, j in enumerate(final):
                name = plot_names[k]
                color = marker_colors[k]
                plt.scatter(t_trunc[j], x_plot[i, j], color=color, s=35, zorder=3)
                plt.text(t_trunc[j] + label_shift, x_plot[i, j], name, color="black", fontsize=9, fontweight = "bold")
                  
            plt.title(rec["id"])
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (V)")
            plt.legend(fontsize=12)
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.show()

            ans = input("y=keep, n=relabel, s=skip, q=quit: ").strip().lower()

            if ans == "q":
                stop_now = True
                break
            if ans == "s":
                continue
            
            if ans == "n":
                plt.figure(figsize=(7, 4))
                plt.plot(t_trunc, x_plot[i], label = "Channel 2",linewidth=2)

                #finds all local extrema and plots with index labels
                for j in range(1, len(x_plot[i]) - 1):
                    peak = x_plot[i, j] > x_plot[i, j - 1] and x_plot[i, j] > x_plot[i, j + 1]
                    trough = x_plot[i, j] < x_plot[i, j - 1] and x_plot[i, j] < x_plot[i, j + 1]
                    if peak:
                        plt.scatter(t_trunc[j], x_plot[i, j], color="red", s=35, zorder=3)
                        plt.text(t_trunc[j] + label_shift, x_plot[i, j], str(j), color="black", fontsize=9, fontweight = "bold")
                    
                    if trough:
                        plt.scatter(t_trunc[j], x_plot[i, j], color="black", s=35, zorder=3)
                        plt.text(t_trunc[j] + label_shift, x_plot[i, j], str(j), color="black", fontsize=9, fontweight = "bold")

                plt.title(rec["id"])
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (V)")
                plt.legend(fontsize=12)
                
                ax = plt.gca()
                ax.spines['top'].set_visible(False) 
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                plt.show()

                #user input for correct labels
                print("enter correct labels: ")
                w1p = input("W1 peak: ")
                w1t = input("W1 trough: ")
                w4p = input("W4 peak: ")
                w4t = input("W4 trough: ")
                w5p = input("W5 peak: ")
                w5t = input("W5 trough: ")
                final = [w1p, w1t, w4p, w4t, w5p, w5t]

                plt.figure(figsize=(7, 4))
                plt.plot(t_trunc, x_plot[i], label = "Channel 2", linewidth=2)
                
                for k, j in enumerate(final):
                    name = plot_names[k]
                    color = marker_colors[k]                    
                    plt.scatter(t_trunc[j], x_plot[i, j], color=color, s=35, zorder=3)
                    plt.text(t_trunc[j] + label_shift, x_plot[i, j], name, color="black", fontsize=9, fontweight = "bold")
                
                plt.title(rec["id"])
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (V)")
                plt.legend(fontsize=12)
                
                ax = plt.gca()
                ax.spines['top'].set_visible(False) 
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                plt.show()

        final = [int(x) for x in final]
        #convert the truncated preaks to the full signal lcoations
        full_idx = [trunc.start + x for x in final]
        peak_times = [float(t_trunc[x]) for x in final]

        prem_vals = [x_prem[i, j] for j in full_idx]
        noise_floor = x_prem[i, noisefloor_range].mean()
        vals_minus_noise = [v - noise_floor for v in prem_vals]
       
        #for csv output
        row = {"Stimulus": rec["stim"], "Rat": rec["rat"], "Day": rec["day"],"SoundLevel": rec["level"].replace("dB", ""),
               "Baseline_noise_floor": noise_floor}
        peak_names = ["W1_Peak", "W1_Trough", "W4_Peak", "W4_Trough", "W5_Peak", "W5_Trough"]

        for peak_number in range(len(peak_names)):
            peak_name = peak_names[peak_number]
            idx_trunc = final[peak_number]
            idx_full = full_idx[peak_number]
            time_ms = peak_times[peak_number]
            amp_premeaned = prem_vals[peak_number]
            amp_minus_noise = vals_minus_noise[peak_number]
    
            row[peak_name + "_idx_trunc"] = idx_trunc
            row[peak_name + "_idx_full"] = idx_full
            row[peak_name + "_ms"] = float(time_ms)
            row[peak_name + "_amp_premeaned"] = float(amp_premeaned)
            row[peak_name + "_amp_minus_noise_floor"] = float(amp_minus_noise)
        amplitude_rows.append(row)

        #store the plotting info to make it easy to do the random plotting check later
        random_plots.append({"title": rec["id"], "t": np.linspace(0, 29.9827, x_prem.shape[1]), "ch2": x_prem[i].copy(), "idx": full_idx})

    if stop_now:
        break
    
if manual_check and not start_found:
    print("start point not found")


#%% save amps as csv

amplitudes = pd.DataFrame(amplitude_rows)
amp_output_file = os.path.join(output_folder, amp_output_name + ".csv")
amplitudes.to_csv(amp_output_file, index=False)

#%% random check

#randomly plots 20 full signals with labeled predictions to check if everything seems okay
    #or however many signals you want it to, just change n
    
n = 20
for p in random.sample(random_plots, n):
    plt.figure(figsize=(8, 4))
    plt.plot(p["t"], p["ch2"], label = "Channel 2", linewidth=2)

    for k, j in enumerate(p["idx"]):
        name = plot_names[k]
        color = marker_colors[k]
        plt.scatter(p["t"][j], p["ch2"][j], color=color, s=35, zorder=3)
        plt.text(p["t"][j] + label_shift, p["ch2"][j], name, color="black", fontsize=9, fontweight = "bold")

    plt.title(p["title"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (V)")
    plt.legend(fontsize=12)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)

    
    plt.tight_layout()
    plt.show()
