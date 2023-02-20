import numpy as np
from scipy.special import expit
import torch
from torch import distributions as dist
from copy import deepcopy
from tqdm import tqdm


def cartesian_product(*arrays):

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def eval_trajectory(
        trajectory,
        current_n_pres, current_delta,
        delays,
        inv_temp,
        threshold,
        initial_forget_rates,
        repetition_rates):

    t_remaining = len(trajectory)

    n_item = len(current_n_pres)

    n_pres = deepcopy(current_n_pres)
    delta = deepcopy(current_delta)

    for item in range(n_item):

        item_pres = trajectory == item
        n_pres_traj = np.sum(item_pres)
        n_pres[item] += n_pres_traj
        if n_pres_traj == 0:
            delta[item] += np.sum(delays)
        else:
            idx_last_pres = np.arange(t_remaining)[item_pres][-1]
            delta[item] = np.sum(delays[idx_last_pres:])

    p = np.zeros(n_item)

    view = n_pres > 0
    rep = n_pres[view] - 1.
    delta = delta[view]

    init_fr = initial_forget_rates[view]
    rep_eff = repetition_rates[view]

    forget_rate = init_fr * (1 - rep_eff) ** rep
    logp_recall = - forget_rate * delta

    p[n_pres > 0] = np.exp(logp_recall)

    learning_reward = np.mean(expit(inv_temp * (p - threshold)))

    return learning_reward


def main():

    n_item = 3

    inv_temp = 50.

    threshold = 0.9

    n_iter_per_session = 2
    time_per_iter = 2
    break_length = 10
    n_session = 2

    t = 0

    t_max = n_session * n_iter_per_session

    t_remaining = t_max - t

    initial_forget_rates = np.ones(n_item) * 0.01
    repetition_rates = np.ones(n_item) * 0.2

    delays = np.tile(
        [time_per_iter
         for _ in range(n_iter_per_session - 1)]
        + [break_length, ],
        n_session)[t:]

    current_n_pres = np.zeros(n_item)
    current_delta = np.ones(n_item)

    print("N possible trajectories:", n_item ** t_remaining)

    all_traj = cartesian_product(*[np.arange(n_item)
                                   for _ in range(t_remaining)])

    results = []

    for trajectory in all_traj:

        learning_reward = eval_trajectory(
                trajectory=trajectory,
                current_n_pres=current_n_pres,
                current_delta=current_delta,
                delays=delays,
                inv_temp=inv_temp,
                threshold=threshold,
                initial_forget_rates=initial_forget_rates,
                repetition_rates=repetition_rates)
        results.append(learning_reward)

    max_r = np.max(results)
    best = results == max_r
    print("Best possible reward", max_r)
    print("N best:", np.sum(best))
    print("Best are:\n", all_traj[best])

    n_epochs = 100
    lr = 0.2
    n_sample = 500

    t_max = n_session * n_iter_per_session

    t_remaining = t_max - t

    logits_action = torch.nn.Parameter(torch.rand((t_remaining, n_item)))

    opt = torch.optim.Adam([logits_action, ], lr=lr)

    p = dist.Categorical(logits=torch.ones((t_remaining, n_item)))

    delays = np.tile(
        [time_per_iter
         for _ in range(n_iter_per_session - 1)]
        + [break_length, ],
        n_session)[t:]

    with tqdm(total=n_epochs, position=1, leave=False) as pbar:
        for epoch in range(n_epochs):

            opt.zero_grad()

            q = dist.Categorical(logits=logits_action)
            trajectories = q.sample((n_sample,))

            loss = 0

            for trajectory in trajectories:

                learning_reward = eval_trajectory(
                    trajectory=trajectory.numpy(),
                    current_n_pres=current_n_pres,
                    current_delta=current_delta,
                    delays=delays,
                    inv_temp=inv_temp,
                    threshold=threshold,
                    initial_forget_rates=initial_forget_rates,
                    repetition_rates=repetition_rates)

                loss -= q.log_prob(trajectory).sum().exp() * learning_reward

            loss += torch.distributions.kl_divergence(q, p).sum()

            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": loss.item()})
            pbar.update()

    traj = np.argmax(logits_action.detach().numpy(), axis=1)

    n_pres = deepcopy(current_n_pres)
    delta = deepcopy(current_delta)

    for item in range(n_item):

        item_pres = traj == item
        n_pres_traj = np.sum(item_pres)
        n_pres[item] += n_pres_traj
        if n_pres_traj == 0:
            delta[item] += np.sum(delays)
        else:
            idx_last_pres = np.arange(t_remaining)[item_pres][-1]
            delta[item] = np.sum(delays[idx_last_pres:])

    p = np.zeros(n_item)

    view = n_pres > 0
    rep = n_pres[view] - 1.
    delta = delta[view]

    init_fr = initial_forget_rates[view]
    rep_eff = repetition_rates[view]

    forget_rate = init_fr * (1 - rep_eff) ** rep
    logp_recall = - forget_rate * delta

    p[n_pres > 0] = np.exp(logp_recall)

    learning_reward = np.mean(expit(inv_temp * (p - threshold)))

    print(traj)
    print(learning_reward)


if __name__ == "__main__":
    main()
