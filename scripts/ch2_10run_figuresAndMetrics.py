# %% imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats


#%% plotting stuff

matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',  '#4DBEEE', '#A2142F', '#008080']
plt.rc('axes', prop_cycle=cycler(color=matlab_colors))
plt.rcParams.update({"font.family": "Arial", "font.size": 16})


# %%

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(os.path.dirname(script_folder))

outputs_folder = os.path.join(base_folder, "outputs")

results_name = "ch2_10runs_20260623_160201"
results_folder = os.path.join(outputs_folder, results_name)

csv_path = os.path.join(results_folder, "ch2_all_trial_stats.csv")

figure_folder = os.path.join(results_folder, "ch2_figures_10runs_sd")
os.makedirs(figure_folder, exist_ok=True)

df = pd.read_csv(csv_path)


stim_labels = {"click": "Click",
               "4k": "4 kHz",
               "8k": "8 kHz",
               "10k": "10 kHz",
               "16k": "16 kHz"}
stimuli = list(stim_labels)

label_names = ["W1 Peak", "W1 Trough", "W4 Peak", "W4 Trough", "W5 Peak", "W5 Trough"]
target_labels = {"target1": "W1 Peak",
                 "target2": "W1 Trough",
                 "target3": "W4 Peak",
                 "target4": "W4 Trough",
                 "target5": "W5 Peak",
                 "target6": "W5 Trough"}

exp_groups = ["Functional", "Sham"]

# df = df[df["group"].isin(exp_groups)].copy()

df["stimulus"] = df["frequency"].map(stim_labels)
df["target_name"] = df["target"].map(target_labels)

plot_levels = sorted(df["sound_level"].unique())
runs = sorted(df["run_number"].unique())

# %% 

accuracy = pd.concat([df.assign(stimulus="Overall", target_name="all targets"),
                      df.assign(stimulus="Overall"), df.assign(target_name="all targets"), df], ignore_index=True)

accuracy = pd.concat([accuracy, accuracy.assign(group="Overall")], ignore_index=True)

acc_by_run = (accuracy.groupby(["run_number", "group", "sound_level", "stimulus", "target_name"],as_index=False).agg(run_accuracy=("is_corr", "mean")))

acc_by_run["run_accuracy"] = acc_by_run["run_accuracy"] * 100

acc_sum = (acc_by_run.groupby(["group", "sound_level", "stimulus", "target_name"], as_index=False).agg(mean_accuracy=("run_accuracy", "mean"), sd_between_runs=("run_accuracy", "std")))

# %% acc plots

#overall
temp = acc_sum[(acc_sum["group"] == "Overall") & (acc_sum["stimulus"] == "Overall") & 
               (acc_sum["target_name"] == "all targets")].sort_values("sound_level")

plt.figure(figsize=(6, 4.5))
plt.errorbar(temp["sound_level"], temp["mean_accuracy"], yerr=temp["sd_between_runs"], fmt="o-", linewidth=2.5,
             markersize=11,capsize=4, capthick=2.5, label="Channel 2")

plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Overall", weight="bold")
plt.xticks(plot_levels)
plt.ylim(67, 103)
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.legend(fontsize = 12, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "overall_accuracy.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()


#ind target
plt.figure(figsize=(6, 4.5))
for target_name in label_names:
    temp = acc_sum[(acc_sum["group"] == "Overall") & (acc_sum["stimulus"] == "Overall") & 
                   (acc_sum["target_name"] == target_name)].sort_values("sound_level")

    plt.errorbar(temp["sound_level"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=target_name)

plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Channel 2 - Overall", weight="bold")
plt.xticks(plot_levels)
plt.ylim(67, 103)
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.legend(fontsize = 12, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "overall_accuracy_by_target.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()


plt.figure(figsize=(6, 4.5))
for st in stimuli:
    st_name = stim_labels[st]
    temp = acc_sum[(acc_sum["group"] == "Overall") & (acc_sum["stimulus"] == st_name) & 
                   (acc_sum["target_name"] == "all targets")].sort_values("sound_level")
    plt.errorbar(temp["sound_level"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=st_name)

plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Channel 2 - Ind Stimuli", weight="bold")
plt.xticks(plot_levels)
plt.grid(False)
plt.legend(fontsize = 12, loc="lower right")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "acc_by_stimulus.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()


for target_name in label_names:
    plt.figure(figsize=(6, 4.5))

    for st in stimuli:
        st_name = stim_labels[st]
        temp = acc_sum[(acc_sum["group"] == "Overall") & (acc_sum["stimulus"] == st_name) & 
                       (acc_sum["target_name"] == target_name)].sort_values("sound_level")

        plt.errorbar(temp["sound_level"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                     markersize=11, capsize=4, capthick=2.5, label=st_name)

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Channel 2 - {target_name}", weight="bold")
    plt.xticks(plot_levels)
    plt.ylim(62, 103)
    plt.grid(False)
    plt.legend(fontsize = 12, loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks(range(65, 101, 5))
    ax.set_yticklabels([str(y) for y in range(65, 101, 5)])
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"accuracy_by_stimulus_{target_name}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()


for st in stimuli:
    st_name = stim_labels[st]
    plt.figure(figsize=(6, 4.5))

    for target_name in label_names:        
        temp = acc_sum[(acc_sum["group"] == "Overall") & (acc_sum["stimulus"] == st_name) &
                       (acc_sum["target_name"] == target_name)].sort_values("sound_level")
        plt.errorbar(temp["sound_level"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                     markersize=11, capsize=4, capthick=2.5, label=target_name)

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Channel 2 - {st_name}", weight="bold")
    plt.xticks(plot_levels)
    plt.grid(False)
    plt.legend(fontsize = 12, loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"accuracy_by_target_{st}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()


# %% exposure group

#overall
plt.figure(figsize=(6, 4.5))
for group_name in exp_groups:    
    temp = acc_sum[(acc_sum["group"] == group_name) & (acc_sum["stimulus"] == "Overall") &
                   (acc_sum["target_name"] == "all targets")].sort_values("sound_level")
    
    plt.errorbar(temp["sound_level"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=group_name)


plt.xlabel("Sound Level (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Channel 2 - Overall", weight="bold")
plt.xticks(plot_levels)
plt.ylim(89.5, 100.5)
plt.grid(False)
plt.legend(fontsize = 12, loc="lower right")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "saf_acc.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()


for st in stimuli:
    st_name = stim_labels[st]
    plt.figure(figsize=(6, 4.5))

    for group_name in exp_groups:
        temp = acc_sum[(acc_sum["group"] == group_name) & (acc_sum["stimulus"] == st_name) &
                       (acc_sum["target_name"] == "all targets")].sort_values("sound_level")

        plt.errorbar(temp["sound_level"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                     markersize=11, capsize=4, capthick=2.5, label=group_name)

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Channel 2 - {st_name}", weight="bold")
    plt.xticks(plot_levels)
    plt.grid(False)
    plt.legend(fontsize = 12, loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"saf_acc_{st}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()


# %% 

move = df[df["is_corr"]].copy()

move_all = pd.concat([move.assign(stimulus="Overall", target_name="all targets"), move.assign(stimulus="Overall"),
                      move.assign(target_name="all targets"), move], ignore_index=True)

move_by_run = (move_all.groupby(["run_number", "sound_level", "stimulus", "target_name"], 
                                as_index=False).agg(n_per_run=("adjustment_ms", "size"), run_mean_ms=("adjustment_ms", "mean"),
                                                    run_median_ms=("adjustment_ms", "median"),within_run_sd_ms=("adjustment_ms", "std")))

move_sum = (move_by_run.groupby(["sound_level", "stimulus", "target_name"], 
                                as_index=False).agg(mean_movement_ms=("run_mean_ms", "mean"), sd_between_runs=("run_mean_ms", "std"), 
                                                    n_runs=("run_number", "nunique"), median_n_per_run=("n_per_run", "median")))


# %% 

temp = move_sum[(move_sum["stimulus"] == "Overall") & (move_sum["target_name"] == "all targets")].sort_values("sound_level")

plt.figure(figsize=(6, 4.5))
plt.errorbar(temp["sound_level"], temp["mean_movement_ms"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
             markersize=11, capsize=4, capthick=2.5)

plt.xlabel("Sound Level (dB)")
plt.ylabel("Prediction Adjustment (ms)")
plt.title("Channel 2 - Overall", weight="bold")
plt.xticks(plot_levels)
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "overall_mvmt.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()


plt.figure(figsize=(6, 4.5))
for target_name in label_names:
    temp = move_sum[(move_sum["stimulus"] == "Overall") & (move_sum["target_name"] == target_name)].sort_values("sound_level")
    plt.errorbar(temp["sound_level"], temp["mean_movement_ms"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=target_name)

plt.xlabel("Sound Level (dB)")
plt.ylabel("Prediction Adjustment (ms)")
plt.title("Channel 2 - Overall", weight="bold")
plt.xticks(plot_levels)
plt.grid(False)
plt.legend(fontsize = 12, loc="upper right")
plt.ylim(-0.01, 0.16)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([0.00, 0.05, 0.10, 0.15])
ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "movement_by_target.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()


#mvt per stim
plt.figure(figsize=(6, 4.5))
for st in stimuli:
    st_name = stim_labels[st]
    temp = move_sum[(move_sum["stimulus"] == st_name) & (move_sum["target_name"] == "all targets")].sort_values("sound_level")

    plt.errorbar(temp["sound_level"], temp["mean_movement_ms"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=st_name)


plt.xlabel("Sound Level (dB)")
plt.ylabel("Prediction Adjustment (ms)")
plt.title("Channel 2 - Ind Stimuli", weight="bold")
plt.xticks(plot_levels)
plt.grid(False)
plt.legend(fontsize = 12, loc="upper right")
plt.ylim(-0.01, 0.16)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([0.00, 0.05, 0.10, 0.15])
ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "movement_by_stim.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()

#mvmt per stim per target
for st in stimuli:
    st_name = stim_labels[st]
    plt.figure(figsize=(6, 4.5))

    for target_name in label_names:
        temp = move_sum[(move_sum["stimulus"] == st_name) & (move_sum["target_name"] == target_name)].sort_values("sound_level")
        
        plt.errorbar(temp["sound_level"], temp["mean_movement_ms"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                     markersize=11, capsize=4, capthick=2.5, label=target_name)

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Prediction Adjustment (ms)")
    plt.title(f"Channel 2 - {st_name}", weight="bold")
    plt.xticks(plot_levels)
    plt.grid(False)
    plt.legend(fontsize = 12, loc="upper right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"movement_by_target_{st}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()


for target_name in label_names:
    plt.figure(figsize=(6, 4.5))

    for st in stimuli:
        st_name = stim_labels[st]
        temp = move_sum[(move_sum["stimulus"] == st_name) & (move_sum["target_name"] == target_name)].sort_values("sound_level")
        plt.errorbar(temp["sound_level"], temp["mean_movement_ms"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                     markersize=11, capsize=4, capthick=2.5, label=st_name)

    plt.xlabel("Sound Level (dB)")
    plt.ylabel("Prediction Adjustment (ms)")
    plt.title(f"Channel 2 - {target_name}", weight="bold")
    plt.xticks(plot_levels)
    plt.grid(False)
    plt.legend(fontsize = 12, loc="upper right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"mvmt_by_stim_{target_name}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()

#%% probability

def mean_edcf(data_by_run, label, x_start):
    run_ecdfs = []
    all_x_vals = []

    for data_vals in data_by_run:
        if len(data_vals) > 0:
            res = stats.ecdf(data_vals)
            run_ecdfs.append(res.cdf)
            all_x_vals += list(res.cdf.quantiles)

    if len(run_ecdfs) == 0:
        return

    # x_vals = np.unique(all_x_vals)
    x_vals = np.unique(np.r_[x_start, all_x_vals])
    cdf_vals = np.asarray([ecdf.evaluate(x_vals) for ecdf in run_ecdfs])

    mean_cdf = np.mean(cdf_vals, axis=0)
    sd_cdf = np.std(cdf_vals, axis=0, ddof=1)

    line = plt.step(x_vals, mean_cdf, where="post", label=label, linewidth=2.5)[0]
    plt.fill_between(x_vals, (mean_cdf - sd_cdf), (mean_cdf + sd_cdf), step="post", color=line.get_color(), alpha=0.15)

#overall prob
x_start = -0.035

for tname in label_names:
    plt.figure(figsize=(6, 4.5))
    plt.axhline(0.9, color="gray", linestyle= "--", linewidth=1.5,label="90% probability")

    for sl in sorted(plot_levels, reverse=True):
        data_by_run = []

        for run in runs:
            data_vals = move[(move["run_number"] == run) & (move["target_name"] == tname) & 
                             (move["sound_level"] == sl)]["adjustment_ms"].dropna().to_numpy()
            data_by_run.append(data_vals)
        mean_edcf(data_by_run, f"{sl} dB", x_start)

    plt.xlabel("Prediction Adjustment (ms)")
    plt.ylabel("Probability")
    plt.title(f"Channel 2 - {tname}", weight="bold")
    # plt.xlim(-0.1, 0.85)
    plt.xlim(-0.05, 0.85)
    plt.ylim(-0.02, 1.02)
    plt.grid(False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize = 12, loc="lower right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(figure_folder, f"prob_overall_{tname}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()


#per stim
for st in stimuli:
    st_name = stim_labels[st]

    plt.figure(figsize=(6, 4.5))
    plt.axhline(0.9, color="gray", linestyle= "--", linewidth=1.5,label="90% probability")

    for sl in sorted(plot_levels, reverse=True):
        data_by_run = []

        for run in runs:
            data_vals = move[(move["run_number"] == run) & (move["frequency"] == st) & 
                             (move["sound_level"] == sl)]["adjustment_ms"].dropna().to_numpy()

            data_by_run.append(data_vals)
        mean_edcf(data_by_run, f"{sl} dB", x_start)

    plt.xlabel("Prediction Adjustment (ms)")
    plt.ylabel("Probability")
    plt.title(f"Channel 2 - {st_name}", weight="bold")
    plt.xlim(-0.05, 0.85)
    plt.ylim(-0.02, 1.02)
    plt.grid(False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize = 12, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"prob_{st}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()
    
    
#per stim and target
for st in stimuli:
    st_name = stim_labels[st]

    for tname in label_names:
        plt.figure(figsize=(6, 4.5))
        plt.axhline(0.9, color="gray", linestyle= "--", linewidth=1.5,label="90% probability")
        
        for sl in sorted(plot_levels, reverse=True):
            data_by_run = []

            for run in runs:
                data_vals = move[(move["run_number"] == run) & (move["frequency"] == st) & (move["target_name"] == tname) &
                                 (move["sound_level"] == sl)]["adjustment_ms"].dropna().to_numpy()
                data_by_run.append(data_vals)
            mean_edcf(data_by_run, f"{sl} dB", x_start)

        plt.xlabel("Prediction Adjustment (ms)")
        plt.ylabel("Probability")
        plt.title(f"Channel 2 - {st_name} - {tname}", weight="bold")
        plt.xlim(-0.05, 0.85)
        plt.ylim(-0.02, 1.02)
        plt.grid(False)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(fontsize = 12, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f"prob_{st}_{tname}.svg"), format="svg", bbox_inches="tight", transparent=True)
        plt.show()
        



# %%
#pooled quantile bins



n_bins = 5
quantiles = np.linspace(0, 1, n_bins + 1)

cal_all = pd.concat([df.assign(stimulus="Overall"), df.assign(target_name="all targets"), df], ignore_index=True)
cal_rows = []

for st_name in cal_all["stimulus"].unique():
    st_df = cal_all[cal_all["stimulus"] == st_name]

    for target_name in st_df["target_name"].unique():
        temp = st_df[st_df["target_name"] == target_name]

        bin_edges = np.quantile(temp["raw_conf"], quantiles)

        for run in runs:
            run_data = temp[temp["run_number"] == run]

            for b in range(n_bins):
                low = bin_edges[b]
                high = bin_edges[b + 1]

                if b == n_bins - 1:
                    in_bin = run_data[(run_data["raw_conf"] >= low) & (run_data["raw_conf"] <= high)]
                else:
                    in_bin = run_data[(run_data["raw_conf"] >= low) & (run_data["raw_conf"] < high)]

                if len(in_bin) > 0:
                    cal_rows.append({"run_number": run,
                                     "stimulus": st_name,
                                     "target_name": target_name,
                                     "bin": b + 1,
                                     "mean_confidence": in_bin["raw_conf"].mean(),
                                     "accuracy": (in_bin["raw_is_corr"].mean() * 100),
                                     "n": len(in_bin)})

cal_by_run = pd.DataFrame(cal_rows)

cal_sum = (cal_by_run.groupby(["stimulus", "target_name", "bin"], as_index=False).agg(mean_confidence=("mean_confidence", "mean"),
                                                                                      mean_accuracy=("accuracy", "mean"),
                                                                                      sd_between_runs=("accuracy", "std"),
                                                                                      n_runs=("run_number", "nunique"),
                                                                                      mean_n_per_run=("n", "mean")))


#all stim per target
plt.figure(figsize=(6, 4.5))
for target_name in label_names:
    temp = cal_sum[(cal_sum["stimulus"] == "Overall") & (cal_sum["target_name"] == target_name)].sort_values("bin")
    plt.errorbar(temp["mean_confidence"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=target_name)

plt.plot([0, 1], [0, 100], linestyle= "--", color = "gray", linewidth=1.5)
plt.title("Channel 2 - Overall", weight="bold")
plt.xlabel("Softmax Probability")
plt.ylabel("Accuracy (%)")
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.legend(fontsize = 12, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "cal_by_target.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()

#all targets per stim
plt.figure(figsize=(6, 4.5))
for st in stimuli:
    st_name = stim_labels[st]    
    temp = cal_sum[(cal_sum["stimulus"] == st_name) & (cal_sum["target_name"] == "all targets")].sort_values("bin")

    plt.errorbar(temp["mean_confidence"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=st_name)

plt.plot([0, 1], [0, 100], linestyle= "--", color="gray", linewidth=1.5)
plt.title("Channel 2 - Ind Stimuli", weight="bold")
plt.xlabel("Softmax Probability")
plt.ylabel("Accuracy (%)")
plt.grid(False)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.legend(fontsize = 12, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "cal_by_stim.svg"), format="svg", bbox_inches="tight", transparent=True)
plt.show()


for st in stimuli:
    st_name = stim_labels[st]
    plt.figure(figsize=(6, 4.5))

    for target_name in label_names:
        temp = cal_sum[(cal_sum["stimulus"] == st_name) & (cal_sum["target_name"] == target_name)].sort_values("bin")
        plt.errorbar(temp["mean_confidence"], temp["mean_accuracy"], yerr=temp["sd_between_runs"],fmt="o-", linewidth=2.5,
                     markersize=11, capsize=4, capthick=2.5, label=target_name)

    plt.plot([0, 1], [0, 100], linestyle= "--", color="gray", linewidth=1.5)
    plt.title(f"Channel 2 - {st_name}", weight="bold")
    plt.xlabel("Softmax Probability")
    plt.ylabel("Accuracy (%)")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize = 12, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"cal_{st}.svg"), format="svg", bbox_inches="tight", transparent=True)
    plt.show()



# %%

#mvmt distance
move_table_by_run = (move_all.groupby(["run_number", "stimulus", "target_name"], 
                                      as_index=False).agg(n_per_run=("adjustment_ms", "size"),
                                                          run_mean_ms=("adjustment_ms", "mean"),
                                                          run_median_ms=("adjustment_ms", "median"), 
                                                          within_run_sd_ms=("adjustment_ms", "std")))
                                                   
df_shift_stats = (move_table_by_run.groupby(["stimulus", "target_name"],
                                            as_index=False).agg(n_runs=("run_number", "nunique"),
                                                                mean_of_run_means_ms=("run_mean_ms", "mean"),
                                                                sd_between_runs_ms=("run_mean_ms", "std"),
                                                                mean_of_run_medians_ms=("run_median_ms", "mean"),
                                                                mean_within_run_sd_ms=("within_run_sd_ms", "mean"),
                                                                median_n_per_run=("n_per_run", "median")))
                                                                
move_runs_tbl = move_table_by_run.pivot(index=["stimulus", "target_name"], columns="run_number", 
                                         values=["n_per_run", "run_mean_ms", "run_median_ms", "within_run_sd_ms"])

move_runs_tbl.columns = [f"{metric}_run_{int(run):02d}" for metric, run in move_runs_tbl.columns]
move_runs_tbl = move_runs_tbl.reset_index()

df_shift_stats = df_shift_stats.merge(move_runs_tbl, on=["stimulus", "target_name"], how="left")

stim_order = ["Overall"] + [stim_labels[st] for st in stimuli]
target_order = label_names + ["all targets"]

df_shift_stats["stimulus"] = pd.Categorical(df_shift_stats["stimulus"], categories=stim_order, ordered=True)
df_shift_stats["target_name"] = pd.Categorical(df_shift_stats["target_name"], categories=target_order,ordered=True)

df_shift_stats = df_shift_stats.sort_values(["stimulus", "target_name"]).reset_index(drop=True)
df_shift_stats = df_shift_stats.rename(columns={"target_name": "target"})

print("time shift stats (ms):")
print(df_shift_stats.round(4).to_string(index=False))


#90th percentile distance
p90_by_run = (move_all.groupby(["run_number", "stimulus", "target_name", "sound_level"],
                               as_index=False).agg(n_per_run=("adjustment_ms", "size"), run_p90_ms=("adjustment_ms", lambda x: x.quantile(0.90))))

df_p90_time = (p90_by_run.groupby(["stimulus", "target_name", "sound_level"],
                                  as_index=False).agg(mean_p90_ms=("run_p90_ms", "mean"),
                                                      sd_between_runs_ms=("run_p90_ms", "std"),
                                                      n_runs=("run_number", "nunique"),
                                                      median_n_per_run=("n_per_run", "median")))

p90_runs_tbl = p90_by_run.pivot(index=["stimulus", "target_name", "sound_level"], columns="run_number",values=["n_per_run", "run_p90_ms"])
p90_runs_tbl.columns = [ f"{metric}_run_{int(run):02d}" for metric, run in p90_runs_tbl.columns]
p90_runs_tbl = p90_runs_tbl.reset_index()

df_p90_time = df_p90_time.merge(p90_runs_tbl,on=["stimulus", "target_name", "sound_level"], how="left")
df_p90_time["stimulus"] = pd.Categorical(df_p90_time["stimulus"],categories=stim_order, ordered=True)

df_p90_time["target_name"] = pd.Categorical(df_p90_time["target_name"], categories=target_order, ordered=True)
df_p90_time = df_p90_time.sort_values(["stimulus", "target_name", "sound_level"]).reset_index(drop=True)

df_p90_time = df_p90_time.rename(columns={"target_name": "target"})

print("90th percentile mvmt probability (ms):")
print(df_p90_time.round(4).to_string(index=False))


#distance from ground truth incorrect predictions only
err = df[~df["is_corr"]].copy()

err_long = pd.concat([err.assign(stimulus="Overall", target_name="all targets"),err.assign(stimulus="Overall"), 
                      err.assign(target_name="all targets"),err], ignore_index=True)

err_table_by_run = (err_long.groupby(["run_number", "stimulus", "target_name"], 
                                     as_index=False).agg(n_per_run=("adjusted_error_ms", "size"),
                                                         run_mean_ms=("adjusted_error_ms", "mean"),
                                                         run_median_ms=("adjusted_error_ms", "median"),
                                                         within_run_sd_ms=("adjusted_error_ms", "std")))

df_err_stats = (err_table_by_run.groupby(["stimulus", "target_name"], 
                                         as_index=False).agg(n_runs=("run_number", "nunique"),
                                                             mean_of_run_means_ms=("run_mean_ms", "mean"),
                                                             sd_between_runs_ms=("run_mean_ms", "std"),
                                                             mean_of_run_medians_ms=("run_median_ms", "mean"),
                                                             mean_within_run_sd_ms=("within_run_sd_ms", "mean"),
                                                             median_n_per_run=("n_per_run", "median")))

err_runs_tbl = err_table_by_run.pivot(index=["stimulus", "target_name"], columns="run_number",
                                       values=["n_per_run", "run_mean_ms", "run_median_ms", "within_run_sd_ms"])

err_runs_tbl.columns = [f"{metric}_run_{int(run):02d}" for metric, run in err_runs_tbl.columns]
err_runs_tbl = err_runs_tbl.reset_index()

df_err_stats = df_err_stats.merge(err_runs_tbl, on=["stimulus", "target_name"], how="left")

df_err_stats["stimulus"] = pd.Categorical(df_err_stats["stimulus"], categories=stim_order, ordered=True)
df_err_stats["target_name"] = pd.Categorical(df_err_stats["target_name"],categories=target_order, ordered=True)

df_err_stats = df_err_stats.sort_values(["stimulus", "target_name"]).reset_index(drop=True)
df_err_stats = df_err_stats.rename(columns={"target_name": "target"})

print("\nincorrect only distance from ground truth")
print(df_err_stats.round(4).to_string(index=False))






# %% accuracy check

df_plot = df[df["sound_level"].isin(plot_levels)].copy()

overall_acc_by_run = (df_plot.groupby("run_number", as_index=False).agg(n_per_run=("is_corr", "size"),
                                                                        n_correct=("is_corr", "sum"),
                                                                        accuracy=("is_corr", "mean")))
overall_acc_by_run["accuracy"] *= 100
overall_acc_by_run["stimulus"] = "Overall"


stim_acc_by_run = (df_plot.groupby(["run_number", "frequency"], as_index=False).agg(n_per_run=("is_corr", "size"),
                                                                                    n_correct=("is_corr", "sum"),
                                                                                    accuracy=("is_corr", "mean")))

stim_acc_by_run["accuracy"] *= 100
stim_acc_by_run["stimulus"] = stim_acc_by_run["frequency"].map(stim_labels)

total_acc_by_run = pd.concat([overall_acc_by_run[["run_number", "stimulus","n_per_run", "n_correct", "accuracy"]],
                              stim_acc_by_run[["run_number", "stimulus", "n_per_run","n_correct", "accuracy"]]], ignore_index=True)

total_acc_sum = (total_acc_by_run.groupby("stimulus", as_index=False).agg(mean_accuracy=("accuracy", "mean"),
                                                                          sd_between_runs=("accuracy", "std"),
                                                                          n_runs=("run_number", "nunique")))

acc_runs_tbl = total_acc_by_run.pivot(index="stimulus", columns="run_number", values=["n_per_run", "n_correct", "accuracy"])

acc_runs_tbl.columns = [f"{metric}_run_{int(run):02d}" for metric, run in acc_runs_tbl.columns]
acc_runs_tbl = acc_runs_tbl.reset_index()

total_acc_sum = total_acc_sum.merge(acc_runs_tbl, on="stimulus", how="left")

total_acc_sum["stimulus"] = pd.Categorical(total_acc_sum["stimulus"], categories=stim_order, ordered=True)
total_acc_sum = total_acc_sum.sort_values("stimulus").reset_index(drop=True)

print("\nOverall accuracy:")
print(total_acc_sum.round(2).to_string(index=False))




# %% save 

df_shift_stats.to_csv(os.path.join(figure_folder, "movement_summary.csv"), index=False)
df_p90_time.to_csv(os.path.join(figure_folder, "p90_summary.csv"), index=False)

df_err_stats.to_csv(os.path.join(figure_folder, "incorrect_error.csv"), index=False)
total_acc_sum.to_csv(os.path.join(figure_folder, "accuracy_summary.csv"), index=False)





















