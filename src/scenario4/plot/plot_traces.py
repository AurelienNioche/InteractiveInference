import os
import numpy as np
from matplotlib import pyplot as plt


def plot_traces(trace, show=False, save=True,
                run_name=""):

    # Extract data
    belief_trace = trace["assistant_belief"]
    # preferred_target = trace["preferred_target"]
    # user_action_trace = np.asarray(trace["user_action"])
    # assistant_action_trace = np.asarray(trace["assistant_action"])
    # targets_positions = np.asarray(trace["targets_positions"]).T

    n_epochs = len(belief_trace)

    # Create figure
    fig, ax = plt.subplots(
        nrows=1, figsize=(15, 8),
        constrained_layout=True)

    im = ax.imshow(
        np.transpose(belief_trace),
        interpolation="nearest",
        aspect="auto",
        cmap="viridis",
        # vmin=0, vmax=1,
    )

    # ax.invert_yaxis()
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax.set_xlim([0, n_epochs])

    ax.set_xlabel("Time")
    ax.set_ylabel("Target")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Assistant belief")

    if save:
        os.makedirs("fig", exist_ok=True)
        if run_name:
            run_name = "_" + run_name
        else:
            run_name = ""
        plt.savefig(f"fig/trace{run_name}.pdf")
    if show:
        plt.show()
