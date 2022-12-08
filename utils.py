import os, sys, pickle, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load model performance from path
def get_performance(metric_path):
    with open(metric_path, "r") as f:
        metric_dict = json.load(f)
    return metric_dict
# Load scaler from path
def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)  
    return scaler
# Organize metrics into DataFrames
def gather_metrics(model_path):
    path = model_path
    maes, rsmes, val_maes = [], [], []
    for model_folder in os.listdir(path):
        print(f"current model = {model_folder}")
        model_val_maes = np.zeros((100, 12))
        model_folder = model_folder + "/"
        #scaler = load_scaler(path + model_folder + model_folder[:-1] + "_scalers.scale")
        for pot_metric_file in os.listdir(path+model_folder):
            if "test_metric.json" in pot_metric_file:
                print("\tfound test metric...")
                metric_dict = get_performance(path + model_folder + pot_metric_file)
                mae_dict, rmse_dict = {"model": model_folder[:-1]}, {"model": model_folder[:-1]}
                for key in metric_dict.keys():
                    if "mean absolute error" in key:
                        mae_dict[key.split("[")[1][:-1]] = metric_dict[key]
                    elif "root mean squared error" in key:
                        rmse_dict[key.split("[")[1][:-1]] = metric_dict[key]
                maes.append(mae_dict)
                rsmes.append(rmse_dict)
                print("\ttest metric logged.")
            elif "val_maes.json" in pot_metric_file:
                print("\tfound val MAEs...")
                metric_dict = get_performance(path + model_folder + pot_metric_file)
                val_mae_dict = {"model": model_folder[:-1]}
                for key in metric_dict.keys():
                    val_mae_dict[key] = np.mean(metric_dict[key])
                val_maes.append(val_mae_dict)
                print("\tval MAEs logged.")
        mae_df, rmse_df, val_maes_df = pd.DataFrame(maes), pd.DataFrame(rsmes), pd.DataFrame(val_maes).T
        val_maes_df = val_maes_df.rename(columns=val_maes_df.iloc[0]).drop(val_maes_df.index[0])
    return mae_df, rmse_df, val_maes_df
# Make bar plot figure
def bar_plot(df, save_name="bar_plot.png", barwidth=0.2):
    # make figure and axes
    fig, ax = plt.subplots(figsize=(15, 6))
    # aesthetic settings for plot
    font = 'Times New Roman'
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 16, 20, 24
    ax.grid(True, linewidth=1.0, color='0.95')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(3.0)
    for tick in ax.get_yticklabels():
        #tick.set_fontname(font)
        tick.set_fontsize(SMALL_SIZE)
    for tick in ax.get_xticklabels():
        #tick.set_fontname(font)
        tick.set_fontsize(SMALL_SIZE)

    model_names = list(df.iloc[:, 0])
    x_labels = list(df.columns[1:])
    rates = df.iloc[:, 1:].to_numpy().T
    for i, n in enumerate(model_names):
        plt.bar(np.arange(len(x_labels))+i*barwidth, rates[:,i], width=barwidth, label=f'{n}')
    plt.xlabel(f"property", fontdict={"size": SMALL_SIZE})
    plt.ylabel(f"min-max scaled metric", fontdict={"size": SMALL_SIZE})
    plt.title(f"Comparison of MAE (for each objective) on the QM9 Dataset", fontdict={"size": BIGGER_SIZE})
    plt.xticks(np.arange(len(x_labels))+(len(model_names)-1)/2*barwidth, x_labels)
    plt.xlim([-0.25*len(model_names), len(x_labels)])
    plt.ylim([0, 1.25*np.ptp(rates)])
    plt.legend(loc='center', bbox_to_anchor=(0.5, np.min(rates)-0.33))
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

def mae_plot(df, save_name="mae_plot.png"):
    # make figure and axes
    fig, ax = plt.subplots(figsize=(15, 6))
    # aesthetic settings for plot
    font = 'Times New Roman'
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 16, 20, 24
    ax.grid(True, linewidth=1.0, color='0.95')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(3.0)
    for tick in ax.get_yticklabels():
        #tick.set_fontname(font)
        tick.set_fontsize(SMALL_SIZE)
    for tick in ax.get_xticklabels():
        #tick.set_fontname(font)
        tick.set_fontsize(SMALL_SIZE)

    model_names = list(df.columns)
    x_labels = np.arange(df.shape[0])
    for i, n in enumerate(model_names):
        plt.plot(x_labels, df[n].tolist(), label=f'{n}', linewidth=2.0)
    plt.title('Comparison of Average Validation MAE\nover 100 Epochs', fontdict={"size": BIGGER_SIZE})
    plt.xlabel('Epoch', fontdict={"size": SMALL_SIZE})
    plt.ylabel('Average Validation MAE (across all objectives)', fontdict={"size": SMALL_SIZE})
    plt.xlim([0, 100])
    plt.ylim([0, 0.03])
    plt.legend()
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()