import numpy as np
import torch

from run.run import run
from plot.plot import plot

from baseline_policies.conservative import Conservative
from baseline_policies.conservative_not_omniscient import ConservativeNotOmniscient
from baseline_policies.leitner import Leitner
from baseline_policies.random import Random

from active_inference.active_inference_omniscient import ActiveTeacher

from environments.teaching import Teaching


def main():

    n_item = 200
    tau = 0.9
    n_session = 6
    break_length = 24 * 60 ** 2
    time_per_iter = 3
    n_iter_session = 100

    forget_rates = np.ones(n_item) * 0.01
    repetition_rates = np.ones(n_item) * 0.20

    env = Teaching(
        tau=tau,
        n_item=n_item,
        n_session=n_session,
        break_length=break_length,
        time_per_iter=time_per_iter,
        n_iter_per_session=n_iter_session,
        initial_forget_rates=forget_rates,
        repetition_rates=repetition_rates)

    policy = ConservativeNotOmniscient(
        env=env,
        bounds=np.asarray([[0.001, 0.03], [0.1, 0.3]]),
        grid_methods=('lin', 'lin'))

    env.reset()
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)

    policy = Conservative(env=env)
    env.reset()
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)

    # prior = torch.zeros((env.t_max, env.n_item))
    # for t in range(env.t_max):
    #     prior[t] = torch.from_numpy(np.eye(env.n_item)[actions[t]])
    #
    # policy = ActiveTeacher(env=env, prior=prior)
    # actions, rewards = run(env=env, policy=policy)
    # plot(actions=actions, rewards=rewards, env=env, policy=policy)

    policy = Leitner(env=env, delay_factor=2, delay_min=3)
    env.reset()
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)

    # policy = Random(env=env)
    # env.reset()
    # actions, rewards = run(env=env, policy=policy)
    # plot(actions=actions, rewards=rewards, env=env, policy=policy)


if __name__ == "__main__":
    main()
