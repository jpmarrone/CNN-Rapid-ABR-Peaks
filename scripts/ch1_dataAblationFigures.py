# %% imports

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler


# %% 

script_folder = Path(__file__).resolve().parent
base_folder = script_folder.parent

save_root = base_folder/"outputs"/"dualChannel_data_forPaper"/"dataset_size_ablation_ch1"

analysis_root = save_root/"analysis_3metrics"
fig_dir = analysis_root/"figures"
table_dir = analysis_root/"tables"

for folder in [analysis_root, fig_dir, table_dir]:
    folder.mkdir(parents=True, exist_ok=True)


strat = "freq_percent"
matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',  '#4DBEEE', '#A2142F', '#008080']
plt.rc("axes", prop_cycle=cycler(color=matlab_colors))
plt.rcParams.update({"font.family": "Arial", "font.size": 16})


# %% 

data_dir = save_root/"data"
summary_dir = save_root/"summary"

summary_list = []

for summary_path in sorted(summary_dir.glob("summary_*.csv")):
    run_id = summary_path.stem.split("summary_", 1)[1]
    data_path = data_dir/f"data_{run_id}.csv"

    if not data_path.exists() or data_path.stat().st_size == 0:
        continue
        
    data_df = pd.read_csv(data_path)
    
    if data_df.empty or data_df.loc[0, "status"] != "completed":
        continue
    
    summary_df = pd.read_csv(summary_path)
    summary_list.append(summary_df)

summary = pd.concat(summary_list, ignore_index=True)
summary = summary[summary["strat"] == strat].copy()
summary = summary[summary["remove_frac"] != 0.975].copy()
summary = summary.sort_values(["remove_frac", "repeat_id"]).reset_index(drop=True)




# %% 

summary_agg = (summary.groupby("remove_frac", as_index=False).agg(initial_mae_ms_mean=("initial_mae_ms", "mean"),
                                                                  initial_mae_ms_sd=("initial_mae_ms", "std"),
                                                                  adjusted_mae_ms_mean=("adjusted_mae_ms", "mean"),
                                                                  adjusted_mae_ms_sd=("adjusted_mae_ms", "std"),
                                                                  adjusted_accuracy_pct_mean=("adjusted_accuracy_pct", "mean"),
                                                                  adjusted_accuracy_pct_sd=("adjusted_accuracy_pct", "std"),
                                                                  n_runs=("repeat_id", "nunique")).sort_values("remove_frac").reset_index(drop=True))
x = summary_agg["remove_frac"] * 100


# %% intial mean abs err

plt.figure(figsize=(7.5, 4.5))
plt.errorbar(x, summary_agg["initial_mae_ms_mean"], yerr=summary_agg["initial_mae_ms_sd"].fillna(0), fmt="o-", linewidth=2.5, 
             markersize=11, capsize=4, capthick=2.5)

plt.xlabel("Dataset Removed (%)")
plt.ylabel("Mean Absolute Error (ms)")
plt.title("Initial Prediction", weight="bold")
plt.xlim(-2, 100)
plt.xticks(range(0, 101, 10))
plt.grid(False)

ax = plt.gca()
ax.set_xticks(range(5, 100, 10), minor=True)
ax.tick_params(axis="x", which="minor", length=2.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(fig_dir/"initial_mae.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()


#adjusted 
plt.figure(figsize=(7.5, 4.5))
plt.errorbar(x, summary_agg["adjusted_mae_ms_mean"], yerr=summary_agg["adjusted_mae_ms_sd"].fillna(0), fmt="o-", linewidth=2.5, 
             markersize=11, capsize=4, capthick=2.5)

plt.xlabel("Dataset Removed (%)")
plt.ylabel("Mean Absolute Error (ms)")
plt.title("Adjusted Prediction", weight="bold")
plt.xlim(-2, 100)
plt.xticks(range(0, 101, 10))
plt.grid(False)

ax = plt.gca()
ax.set_xticks(range(5, 100, 10), minor=True)
ax.tick_params(axis="x", which="minor", length=2.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(fig_dir/"adjusted_mae.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()


#accuracy
plt.figure(figsize=(7.5, 4.5))
plt.errorbar(x, summary_agg["adjusted_accuracy_pct_mean"], yerr=summary_agg["adjusted_accuracy_pct_sd"].fillna(0), fmt="o-", linewidth=2.5, 
             markersize=11, capsize=4, capthick=2.5)

plt.xlabel("Dataset Removed (%)")
plt.ylabel("Accuracy (%)")
plt.title("Adjusted Prediction", weight="bold")
plt.xlim(-2, 100)
plt.xticks(range(0, 101, 10))
plt.grid(False)

ax = plt.gca()
ax.set_xticks(range(5, 100, 10), minor=True)
ax.tick_params(axis="x", which="minor", length=2.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(fig_dir/"adjusted_accuracy.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()



# %% per target

target_col = "target"

target_labels = {"target1": "W1 Peak",
                 "target2": "W1 Trough",
                 "target3": "W3 Peak",
                 "target4": "W3 Trough"}

targets_to_plot = list(target_labels.keys())

summary_target = (summary[summary[target_col].isin(targets_to_plot)].groupby(["remove_frac", target_col], as_index=False)
                  .agg(initial_mae_ms_mean=("initial_mae_ms", "mean"),
                       initial_mae_ms_sd=("initial_mae_ms", "std"),
                       adjusted_mae_ms_mean=("adjusted_mae_ms", "mean"),
                       adjusted_mae_ms_sd=("adjusted_mae_ms", "std"),
                       adjusted_accuracy_pct_mean=("adjusted_accuracy_pct", "mean"),
                       adjusted_accuracy_pct_sd=("adjusted_accuracy_pct", "std"),
                       n_runs=("repeat_id", "nunique")).sort_values(["remove_frac", target_col]).reset_index(drop=True))


# %%

#mae initial
plt.figure(figsize=(7.5, 4.5))

for target in targets_to_plot:
    temp = summary_target[summary_target[target_col] == target].sort_values("remove_frac")
    
    x = temp["remove_frac"] * 100

    plt.errorbar(x, temp["initial_mae_ms_mean"], yerr=temp["initial_mae_ms_sd"].fillna(0), fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=target_labels[target])

plt.xlabel("Dataset Removed (%)")
plt.ylabel("Mean Absolute Error (ms)")
plt.title("Initial Prediction", weight="bold")
plt.xlim(-2, 97)
plt.ylim(-0.05, 0.65)
plt.xticks(range(0, 91, 10))
plt.grid(False)

ax = plt.gca()
ax.set_xticks(range(5, 100, 10), minor=True)
ax.tick_params(axis="x", which="minor", length=2.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(fig_dir/"initial_mae_by_target.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()



#mae adjusted
plt.figure(figsize=(7.5, 4.5))

for target in targets_to_plot:
    temp = summary_target[summary_target[target_col] == target].sort_values("remove_frac")
    x = temp["remove_frac"] * 100

    plt.errorbar(x, temp["adjusted_mae_ms_mean"], yerr=temp["adjusted_mae_ms_sd"].fillna(0), fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=target_labels[target])

plt.xlabel("Dataset Removed (%)")
plt.ylabel("Mean Absolute Error (ms)")
plt.title("Adjusted Prediction", weight="bold")
plt.xlim(-2, 97)
plt.ylim(-0.05, 0.45)
plt.xticks(range(0, 91, 10))
plt.grid(False)

ax = plt.gca()
ax.set_xticks(range(5, 100, 10), minor=True)
ax.tick_params(axis="x", which="minor", length=2.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(fig_dir/"adjusted_mae_by_target.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()



#accuracy
plt.figure(figsize=(7.5, 4.5))

for target in targets_to_plot:
    temp = summary_target[summary_target[target_col] == target].sort_values("remove_frac")
    x = temp["remove_frac"] * 100

    plt.errorbar(x, temp["adjusted_accuracy_pct_mean"], yerr=temp["adjusted_accuracy_pct_sd"].fillna(0), fmt="o-", linewidth=2.5,
                 markersize=11, capsize=4, capthick=2.5, label=target_labels[target])

plt.xlabel("Dataset Removed (%)")
plt.ylabel("Accuracy (%)")
plt.title("Adjusted Prediction", weight="bold")
plt.xlim(-2, 97)
plt.ylim(64, 101)
plt.xticks(range(0, 91, 10))
plt.grid(False)

ax = plt.gca()
ax.set_xticks(range(5, 100, 10), minor=True)
ax.tick_params(axis="x", which="minor", length=2.5)
ax.set_yticks(range(65, 101, 5))
ax.set_yticklabels([str(y) for y in range(65, 101, 5)])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(fig_dir/"adjusted_acc_by_target.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()












