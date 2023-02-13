import numpy as np
import torch

from plot.plot import plot
from run.run import run

from baseline_policies.conservative import Conservative

from active_inference.active_inference import ActiveTeacher

from environments.teaching import Teaching


def main():

    n_item = 200
    tau = 0.9
    n_session = 6
    break_length = 24 * 60 ** 2
    time_per_iter = 3
    n_iter_session = 100

    forget_rates = np.ones(n_item) * 0.02
    repetition_rates = np.ones(n_item) * 0.20

    # n_items = 20
    # tau = 0.9
    # n_session = 6
    # break_length = 10
    # time_per_iter = 3
    # n_iter_session = 10
    #
    # forget_rates = np.ones(n_items) * 0.005
    # repetition_rates = np.ones(n_items) * 0.20

    env = Teaching(
        tau=tau,
        n_item=n_item,
        n_session=n_session,
        break_length=break_length,
        time_per_iter=time_per_iter,
        n_iter_per_session=n_iter_session,
        initial_forget_rates=forget_rates,
        repetition_rates=repetition_rates)

    policy = Conservative(env=env)
    env.reset()
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)

    prior = torch.zeros((env.t_max, env.n_item))
    for t in range(env.t_max):
        prior[t] = torch.from_numpy(np.eye(env.n_item)[traces.actions[t]])

    policy = ActiveTeacher(env=env, prior=prior)
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)


if __name__ == "__main__":
    main()
