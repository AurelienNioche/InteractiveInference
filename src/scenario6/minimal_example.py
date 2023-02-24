
import numpy as np
from scipy.special import expit
import torch
from torch import distributions as dist
from tqdm import tqdm
from types import SimpleNamespace


def cartesian_product(*arrays):

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def eval_trajectory(trajectory, env):

    rng = np.random.default_rng(seed=0)
    rng.random.choice()

    t_remaining = len(trajectory)

    n_item = len(env.n_pres)

    n_pres = env.n_pres.copy()
    delta = env.delta.copy()

    delays = env.delays[-t_remaining:]

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

    init_fr = env.initial_forget_rates[view]
    rep_eff = env.repetition_rates[view]

    forget_rate = init_fr * (1 - rep_eff) ** rep
    logp_recall = - forget_rate * delta

    p[n_pres > 0] = np.exp(logp_recall)

    learning_reward = np.mean(expit(env.reward_inv_temp * (p - env.threshold)))

    n_learnt = np.sum(logp_recall > np.log(env.threshold))
    return learning_reward, n_learnt


def conservative(env):

    log_tau = np.log(env.threshold)

    def threshold_select(n_pres, delta):
        seen = n_pres > 0
        num_item, n_seen = len(seen), np.sum(seen)
        if not n_seen:
            item_to_show = 0
        else:
            log_p_success = cp_log_p_seen(n_pres=n_pres, delta=delta)
            if n_seen == num_item or np.min(log_p_success) <= log_tau:
                item_to_show = np.flatnonzero(seen)[np.argmin(log_p_success)]
            else:
                item_to_show = n_seen  # Present new item
        return item_to_show

    def cp_log_p_seen(n_pres, delta):

        view = n_pres > 0
        rep = n_pres[view] - 1.
        delta = delta[view]

        init_fr = env.initial_forget_rates[np.nonzero(view)]
        rep_eff = env.repetition_rates[np.nonzero(view)]

        forget_rate = init_fr * (1 - rep_eff) ** rep
        logp_recall = - forget_rate * delta
        return logp_recall

    def step(item, n_pres, delta, delay):

        # increase delta
        delta += delay
        # ...specific for item_to_show shown
        delta[item] = delay
        # increment number of presentation
        n_pres[item] += 1

        return n_pres, delta

    min_n_item = 1
    max_n_item = env.n_item

    to_test = np.ones(env.n_item+1, dtype=bool)
    to_test[0] = 0

    # Find the maximum number of item such that every item presented is learnable
    while True:

        n_item = np.random.choice(np.arange(env.n_item+1)[to_test])

        roll_n_pres = env.n_pres[:n_item].copy()
        roll_delta = env.delta[:n_item].copy()

        traj = np.zeros(env.t_max - env.t, dtype=int)

        # Do rollouts...
        for t in range(env.t, env.t_max):  # Already selected first item
            roll_item = threshold_select(n_pres=roll_n_pres, delta=roll_delta)
            roll_n_pres, roll_delta = step(item=roll_item, n_pres=roll_n_pres, delta=roll_delta, delay=env.delays[t])
            traj[t] = roll_item

        log_p_seen = cp_log_p_seen(delta=roll_delta, n_pres=roll_n_pres)
        n_learnt = np.sum(log_p_seen > log_tau)
        if n_learnt < n_item:  # Failure
            max_n_item = n_item - 1
            if max_n_item < min_n_item:
                break         # Solution is min_n_item
        else:                 # Success
            min_n_item = n_item

        if min_n_item == max_n_item:
            break            # Solution is either
        else:
            # Restrict search space
            to_test[n_item] = 0
            to_test[:min_n_item] = 0
            to_test[max_n_item + 1:] = 0

    return traj


def full_explo(env):

    n_traj = env.n_item ** (env.t_max - env.t)

    all_traj = cartesian_product(*[
        np.arange(env.n_item) for _ in range(env.t_max - env.t)])

    results = np.zeros((n_traj, 2))

    for i in tqdm(range(n_traj)):
        traj = all_traj[i]
        learning_reward, n_learnt = eval_trajectory(env=env, trajectory=traj)
        results[i] = learning_reward, n_learnt

    max_r = np.max(results[:, 0])
    best = results[:, 0] == max_r
    print(f"Best possible reward: {max_r} [N learnt = {np.mean(results[best, 1])}]")
    print("N best:", np.sum(best))
    print("Best are:\n", all_traj[best])


def using_gd(
        env,
        n_epochs=100,
        lr=0.2,
        n_sample=100,
        kl_div_factor=0.001):

    t_remaining = env.t_max - env.t
    n_item = env.n_item

    logits_action = torch.nn.Parameter(torch.rand((t_remaining, n_item)))

    opt = torch.optim.Adam([logits_action, ], lr=lr)

    p = dist.Categorical(logits=torch.ones((t_remaining, n_item)))

    with tqdm(total=n_epochs, position=1, leave=False) as pbar:
        for epoch in range(n_epochs):

            opt.zero_grad()

            q = dist.Categorical(logits=logits_action)
            trajectories = q.sample((n_sample,))

            loss, p_sum = 0, 0

            for traj in trajectories:

                learning_reward, n_learnt = eval_trajectory(trajectory=traj.numpy(), env=env)

                p_traj = q.log_prob(traj).sum().exp()
                loss -= p_traj * learning_reward
                p_sum += p_traj

            loss /= p_sum

            kl_div = dist.kl_divergence(q, p).sum()
            loss += kl_div_factor * kl_div

            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": loss.item()})
            pbar.update()

    traj = np.argmax(logits_action.detach().numpy(), axis=1)
    return traj


def main():

    n_item = 4

    env = SimpleNamespace(
        initial_forget_rates=np.ones(n_item) * 0.01,
        repetition_rates=np.ones(n_item) * 0.2,
        threshold=0.9,
        n_iter_per_session=2,
        time_per_iter=2,
        break_length=10,
        n_session=10,
        reward_inv_temp=50.,
        n_item=n_item)

    env.t_max = env.n_session * env.n_iter_per_session
    env.delays = np.tile(
        [env.time_per_iter
         for _ in range(env.n_iter_per_session - 1)]
        + [env.break_length, ],
        env.n_session)

    env.n_pres = np.zeros(n_item)
    env.delta = np.ones(n_item)
    env.t = 0

    n_traj = n_item ** (env.t_max - env.t)
    print("N possible trajectories:", n_traj)

    if n_traj < 10000:
        full_explo(env=env)

    traj = using_gd(env=env)
    learning_reward = eval_trajectory(trajectory=traj, env=env)
    print("GD: Best traj", traj)
    print("GD: Reward", learning_reward)

    traj = conservative(env=env)
    learning_reward = eval_trajectory(trajectory=traj, env=env)
    print("Conservative", traj)
    print("Conservative", learning_reward)


if __name__ == "__main__":
    main()
