import os
import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_traces(trace,
                show=False,
                run_name=""):

    user_goal = trace["user_goal"]
    belief = np.asarray(trace["assistant_belief"])
    assistant_action = np.asarray(trace["assistant_action"])
    n_targets = len(user_goal)
    n_steps = len(belief)

    # Create figure
    fig, axes = plt.subplots(figsize=(15, 8), nrows=n_targets + 1,
                             sharex='row',
                             constrained_layout=True)

    for target in range(n_targets):

        ax = axes[target]

        x = torch.linspace(0., 0.999, 100)

        belief_target = []

        for step in range(n_steps):
            mu, log_var = torch.from_numpy(belief[step, target, :])
            std = torch.exp(0.5 * log_var)
            x_scaled = torch.logit(x)
            belief_step = torch.distributions.Normal(mu, std).log_prob(x_scaled).exp().numpy()
            belief_target.append(belief_step)

        x_min, x_max = 0, n_steps
        y_min, y_max = 1.0, 0.0  # Note the inversion
        im = ax.imshow(
            np.transpose(belief_target),
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
            aspect="auto",
            cmap="viridis",
            # vmin=0, vmax=20.0,
        )

        ax.axhline(user_goal[target], ls="--", color="red")

        ax.invert_yaxis()
        ax.yaxis.get_major_locator().set_params(integer=True)

        ax.set_xlim((0, n_steps))

        ax.set_ylabel("Position")

        ax.set_title(f"target {target}")

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Assistant belief")

    ax = axes[n_targets]

    ax.plot(assistant_action)
    for target in range(n_targets):
        ax.axhline(user_goal[target], ls=":", color=f"C{target}", label=f"target {target}")

    ax.set_xlim((0, n_steps))
    ax.set_yticks([0, 1])

    ax.set_ylabel("Position")

    axes[-1].set_xlabel("Time")

    ax.legend()

    os.makedirs("fig", exist_ok=True)
    if run_name:
        run_name = "_" + run_name
    plt.savefig(f"fig/trace{run_name}.pdf")
    if show:
        plt.show()
