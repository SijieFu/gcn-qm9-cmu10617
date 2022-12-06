import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load model performance from path
def get_performance(metric_path):
    with open(metric_path, "r") as f:
        metric_dict = json.load(f)
    return metric_dict

# Organize metrics into DataFrames
def gather_metrics(model_path):
    path = model_path
    maes = []
    rsmes = []
    for pot_metric_file in os.listdir(path):
        if "test_metric.json" in pot_metric_file:
            metric_file = pot_metric_file
            metric_dict = get_performance(path + metric_file)
            mae_dict = {"model": "_".join(metric_file.split("_")[:-2])}
            rmse_dict = {"model": "_".join(metric_file.split("_")[:-2])}
            for key in metric_dict.keys():
                if "mean absolute error" in key:
                    mae_dict[key.split("[")[1][:-1]] = metric_dict[key]
                elif "root mean squared error" in key:
                    rmse_dict[key.split("[")[1][:-1]] = metric_dict[key]
            maes.append(mae_dict)
            rsmes.append(rmse_dict)
    mae_df = pd.DataFrame(maes)
    rmse_df = pd.DataFrame(rsmes)
    return mae_df, rmse_df

# Make bar plot figure
def bar_plot(df, save_name="test.png", barwidth=0.2):
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
    plt.xlabel(f"property")
    plt.ylabel(f"min-max scaled metric")
    plt.title(f"Comparison of different model architectures\nfor property prediction on the QM9 Dataset")
    plt.xticks(np.arange(len(x_labels))+(len(model_names)-1)/2*barwidth, x_labels)
    plt.xlim([-0.1*len(model_names), len(x_labels)+0.02])
    plt.ylim([np.min(rates)-0.1, 1.0+0.1*np.ptp(rates)])
    plt.legend(loc='center', bbox_to_anchor=(0.5, np.min(rates)-0.33))
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

