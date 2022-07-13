import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

try:
    from assistants.ai_assistant import AiAssistant
    from users.user import User

except ModuleNotFoundError:  # For documentation
    # noinspection PyUnresolvedReferences
    from scenario2.assistants.ai_assistant import AiAssistant
    # noinspection PyUnresolvedReferences
    from scenario2.users.user import User


def plot_traces(trace, show=False):

    # Extract data
    b_trace = trace["b"]
    user_action_trace = np.asarray(trace["user_action"])
    assistant_action_trace = np.asarray(trace["assistant_action"])
    x = np.asarray(trace["x"]).T
    n_epochs = len(b_trace)

    # Create figure
    fig, axes = plt.subplots(nrows=2, figsize=(15, 8),
                             constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(
        np.transpose(b_trace),
        interpolation="nearest",
        aspect="auto",
        # vmin=0, vmax=1,
        cmap="viridis")
    # psi = np.asarray(psi_trace)
    epochs = np.arange(n_epochs)
    # a_trace.append(0)
    a = np.asarray(user_action_trace)
    idx = a == 0
    ax.scatter(epochs[idx], assistant_action_trace[idx], c="none", marker='^', edgecolor="C1")
    idx = a == 1
    ax.scatter(epochs[idx], assistant_action_trace[idx], c='red', marker='v')
    # idx = a==0
    # ax.scatter(epochs[idx], psi[idx], c = c[idx], marker = 'o')
    ax.invert_yaxis()
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax.set_xlim([0, n_epochs])

    ax.set_xlabel("Time")
    ax.set_ylabel("Target")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Assistant belief")

    ax = axes[1]
    for i, x_ in enumerate(x):
        ax.plot(x_, label=f"Target {i}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Distance to the user")

    ax.set_xlim([0, n_epochs])

    ax.legend()

    # plt.tight_layout()

    os.makedirs("fig", exist_ok=True)
    plt.savefig("fig/trace.pdf")
    if show:
        plt.show()


def run(
    seed: int = 1234,
    goal: int = 3,
    n_targets: int = 5,
    step_size: float = 0.01,
    step_size_other: [float, None] = None,  # 0.01 / (n_targets - 1)
    learning_rate: float = 0.01,
    max_n_step: int = 200,
    debug: bool = True,
    user_parameters: [tuple, list, float, None] = None,
    decision_rule: str = "active_inference",
) -> dict:

    assert goal < n_targets

    np.random.seed(seed)
    torch.manual_seed(seed)

    trace = {k: [] for k in ("b", "user_action", "x", "assistant_action")}

    user = User(n_targets=n_targets, parameters=user_parameters, debug=debug)

    assistant = AiAssistant(
        user_model=user,
        step_size=step_size,
        step_size_other=step_size_other,
        n_targets=n_targets,
        learning_rate=learning_rate,
        decision_rule=decision_rule,
        debug=debug)

    _ = user.reset()
    user.goal = goal
    assistant_output = assistant.reset()

    iter_ = range(max_n_step)
    if not debug:
        iter_ = tqdm(iter_)

    for _ in iter_:

        trace["x"].append(assistant_output)
        trace["b"].append(assistant.belief)

        if debug:
            print("Positions", assistant.x)

        user_output, _, user_done, _ = user.step(assistant_output)
        if user_done:
            break

        if debug:
            print("User action", user_output)

        assistant_output, _, assistant_done, _ = assistant.step(user_output)
        if assistant_done:
            break

        trace["user_action"].append(user_output)
        trace["assistant_action"].append(assistant.a)

        if debug:
            print()

    # trace["x"].append(assistant_output)
    # trace["b"].append(assistant.variational_density(b).numpy())

    # print(trace["b"])
    return trace


def main():

    trace = run(debug=False,
                seed=1,
                user_parameters=(8.0, 0.5),
                max_n_step=500,
                n_targets=10,
                step_size=0.01,
                step_size_other=None,
                decision_rule='active_inference',
                )
    plot_traces(trace, show=True)


if __name__ == '__main__':
    main()
