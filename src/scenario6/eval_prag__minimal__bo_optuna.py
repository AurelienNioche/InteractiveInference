import numpy as np
from scipy.special import expit
import torch
from torch import distributions as dist
from copy import deepcopy
from tqdm import tqdm
import optuna


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

    trajectory = np.asarray(trajectory)

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


def find_best_full_explo(
        t_remaining,
        current_n_pres,
        current_delta,
        delays,
        inv_temp,
        threshold,
        initial_forget_rates,
        repetition_rates):

    n_item = len(current_n_pres)

    print("N possible trajectories:", n_item ** t_remaining)

    all_traj = cartesian_product(
        *[np.arange(n_item)
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
        print(trajectory, learning_reward)

    max_r = np.max(results)
    best = results == max_r
    print("Best possible reward", max_r)
    print("N best:", np.sum(best))
    print("Best are:\n", all_traj[best])


def black_box_function(trial, t_remaining, n_item, n_sample,
                       current_n_pres, current_delta, delays, inv_temp, threshold,
                       initial_forget_rates, repetition_rates, kl_div_factor):
    # trial.suggest_int(n_item)
    logits_action = torch.zeros((t_remaining, n_item))
    for i in range(n_item):
        for t in range(t_remaining):
            logits_action[t, i] = trial.suggest_float(f"{(t, i)}", -10.0, 10.0)

    p = dist.Categorical(logits=torch.ones((t_remaining, n_item)))
    q = dist.Categorical(logits=logits_action)
    trajectories = q.sample((n_sample,))

    loss, p_sum = 0, 0

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

        p_traj = q.log_prob(trajectory).sum().exp()
        loss -= p_traj * learning_reward
        p_sum += p_traj

    loss /= p_sum

    kl_div = dist.kl_divergence(q, p).sum()
    loss += kl_div_factor * kl_div

    # reward -= torch.distributions.kl_divergence(q, p).sum()
    return loss.item()


def main():

    # n_item = 2
    #
    # inv_temp = 100.
    #
    # threshold = 0.9
    #
    # n_iter_per_session = 3
    # time_per_iter = 10
    # break_length = 10
    # n_session = 1
    #
    # initial_forget_rates = np.ones(n_item) * 0.01
    # repetition_rates = np.ones(n_item) * 0.2

    n_item = 3

    inv_temp = 100.

    kl_div_factor = 0.001

    threshold = 0.9

    n_iter_per_session = 4
    time_per_iter = 10
    break_length = 10
    n_session = 1

    initial_forget_rates = np.ones(n_item) * 0.01
    repetition_rates = np.ones(n_item) * 0.2

    t = 0

    t_max = n_session * n_iter_per_session

    t_remaining = t_max - t

    delays = np.tile(
        [time_per_iter
         for _ in range(n_iter_per_session - 1)]
        + [break_length, ],
        n_session)[t:]

    current_n_pres = np.zeros(n_item)
    current_delta = np.ones(n_item)

    # -------------------------- #

    find_best_full_explo(
        t_remaining=t_remaining,
        current_n_pres=current_n_pres,
        current_delta=current_delta,
        delays=delays,
        inv_temp=inv_temp,
        threshold=threshold,
        initial_forget_rates=initial_forget_rates,
        repetition_rates=repetition_rates)

    # ------------------------ #

    n_sample = 100

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: black_box_function(
            trial,
            t_remaining=t_remaining, n_item=n_item,
            n_sample=n_sample,
            current_n_pres=current_n_pres,
            current_delta=current_delta,
            delays=delays,
            inv_temp=inv_temp,
            threshold=threshold,
            initial_forget_rates=initial_forget_rates,
            repetition_rates=repetition_rates,
            kl_div_factor=kl_div_factor),
        n_trials=1000,
        show_progress_bar=True)

    print("Estimated best score", study.best_value)
    best_traj = []
    for t in range(t_remaining):
        choice = np.argmax([study.best_params[f'{(t, i)}'] for i in range(n_item)])
        best_traj.append(choice)
    print("best traj", best_traj)
    actual_score = eval_trajectory(
        trajectory=best_traj,
        current_n_pres=current_n_pres,
        current_delta=current_delta,
        delays=delays,
        inv_temp=inv_temp,
        initial_forget_rates=initial_forget_rates,
        repetition_rates=repetition_rates,
        threshold=threshold)
    print("Actual score", actual_score)


if __name__ == "__main__":
    main()
