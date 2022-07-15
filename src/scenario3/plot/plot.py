import os
import numpy as np
from matplotlib import pyplot as plt


def plot_traces(trace,
                show=False,
                run_name=""):

    # Extract data
    user_goal = trace["user_goal"]
    belief_trace = trace["assistant_belief"]
    user_action_trace = np.asarray(trace["user_action"])
    assistant_action_trace = np.asarray(trace["assistant_action"])
    targets_position = np.asarray(trace["targets_position"]).T
    n_epochs = len(belief_trace)

    # Create figure
    fig, axes = plt.subplots(
        nrows=2, figsize=(15, 8),
        constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(
        np.transpose(belief_trace),
        interpolation="nearest",
        aspect="auto",
        cmap="viridis",
        # vmin=0, vmax=1,
    )

    epochs = np.arange(n_epochs)

    a = np.asarray(user_action_trace)
    idx = a == 0
    ax.scatter(epochs[idx], assistant_action_trace[idx], c="none", marker='^', edgecolor="C1")
    idx = a == 1
    ax.scatter(epochs[idx], assistant_action_trace[idx], c='red', marker='v')

    ax.invert_yaxis()
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax.set_xlim([0, n_epochs])

    ax.set_xlabel("Time")
    ax.set_ylabel("Target")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Assistant belief")

    ax = axes[1]
    for i, x in enumerate(targets_position):
        if i == preferred_target:
            lw = 2
        else:
            lw = 1

        ax.plot(x, label=f"Target {i}", lw=lw)

    ax.set_xlabel("Time")
    ax.set_ylabel("Distance to the user")

    ax.set_xlim([0, n_epochs])

    ax.legend()

    os.makedirs("fig", exist_ok=True)
    if run_name:
        run_name = "_" + run_name
    plt.savefig(f"fig/trace{run_name}.pdf")
    if show:
        plt.show()
