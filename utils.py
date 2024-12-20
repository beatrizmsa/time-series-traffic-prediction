import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def boxplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(data=data, y=col, ax=axes[i],palette='pastel')
        axes[i].set_title(col)
        axes[i].set_ylabel("")

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()

def barplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        label = data.groupby(col).size()
        sns.barplot(x=label.index, y=label.values, ax=axes[i],palette='pastel')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].tick_params(axis="x", rotation=90)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()

def boxplot_by_type_visualization(data, columns, title=''):
    columns = [col for col in columns if col != 'traffic_volume']
    n_cols = min(2, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(x='traffic_volume', y=col, data=data, ax=axes[i],palette='pastel')
        axes[i].set_xlabel('traffic_volume')
        axes[i].set_ylabel(col)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()

def scatter_by_type_visualization(data, columns, title=''):
    columns = [col for col in columns if col != 'traffic_volume']
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.scatterplot(x=col, y='traffic_volume', data=data,ax=axes[i],palette='pastel')
        axes[i].set_ylabel('traffic_volume')
        axes[i].set_xlabel(col)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def barplot_by_type_visualization(data, columns, title=''):
    columns = [col for col in columns if col != 'traffic_volume']
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.barplot(x=col, y='traffic_volume', data=data,ax=axes[i],palette='pastel')
        axes[i].set_xlabel('traffic_volume')
        axes[i].set_ylabel(col)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def slideWindow(df, window_in, horizon):
    d = df.values
    X, y = [], []
    idx = df.index[:-window_in]
    for start in range(len(df)-window_in):
        end = start + window_in
        out = end + horizon
        X.append(d[start:end].reshape(-1))
        y.append(d[end:out].ravel())
    cols_x = [f'x{i}' for i in range(1, window_in+1)]
    cols_y = [f'y{i}' for i in range(1, horizon+1)]
    df_xs = pd.DataFrame(X, index=idx, columns=cols_x)
    df_y = pd.DataFrame(y, index=idx, columns=cols_y)
    return pd.concat([df_xs, df_y], axis=1).dropna()


def add_results(data_results, model_name, steps, mae, mse, r2, rmse):
    new_row = {
        "Model": model_name,
        "Horizon": steps,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "RMSE": rmse,
    }
    data_results.loc[len(data_results)] = new_row