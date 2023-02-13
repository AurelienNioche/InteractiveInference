import numpy as np
import matplotlib.pyplot as plt


def plot(traces, env, policy):

    actions, rewards, inf_param = traces.actions, traces.rewards, traces.inferred_parameters

    title = policy.__class__.__name__.lower()

    n_learned = np.array(rewards) * env.n_item

    fig, ax = plt.subplots()
    ax.plot(n_learned)
    ax.set_xlabel("time")
    ax.set_ylabel("n learned")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(actions)), actions, alpha=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    is_item_specific = len(np.unique(env.initial_forget_rates)) > 1
    if not is_item_specific and inf_param:
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        true_fr = np.unique(env.initial_forget_rates)
        true_rep = np.unique(env.repetition_rates)
        inf_fr = [inf_param[t][:, 0] for t in range(len(inf_param))]
        inf_rep = [inf_param[t][:, 1] for t in range(len(inf_param))]

        ax1.plot(inf_fr)
        ax1.axhline(y=true_fr, ls="--", color="black", alpha=0.3)

        ax2.plot(inf_rep, color="C1")
        ax2.axhline(y=true_rep, ls="--", color="black", alpha=0.3)

        ax.set_title(title)
        plt.tight_layout()
        plt.show()
