import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def barplot(x, y, hue, xlabel, ylabel, title, figname):
    """
    Saves a bar chart as figname.png
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(x=x, y=y, hue=hue, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.savefig(f"./figures/{figname}.png", dpi=200)

def lineplot(x, y, hue, xlabel, ylabel, title, figname):
    """
    Saves a line chart as figname.png
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.lineplot(x=x, y=y, hue=hue, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title) 

    plt.savefig(f"./figures/{figname}.png", dpi=200)

def prediction_plot(x, y_true, y_pred, xlabel, ylabel, title, figname):
    """
    Plots true values and predicted values over time and shows the residual between them. Saves as figname.png
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    ax.vlines(x, y_pred, y_true, color="gray", alpha=0.5, label="Residual")
    sns.lineplot(x=x, y=y_true, color="black", label="True", ax=ax)
    sns.lineplot(x=x, y=y_pred, color="tab:red", label="LSTM prediction", ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.savefig(f"./figures/{figname}.png", dpi=200) 