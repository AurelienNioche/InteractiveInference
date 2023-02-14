import numpy as np
import torch

from plot.plot import plot
from run.run import run

from baseline_policies.conservative import Conservative
from baseline_policies.leitner import Leitner
from baseline_policies.random import Random

from active_inference.active_inference_pragmatic_only import ActivePragmaticOnly

from environments.teaching import Teaching


def main():

    n_item = 10
    tau = 0.9
    n_session = 20
    time_per_iter = 3
    break_length = time_per_iter*40 # 24 * 60 ** 2
    n_iter_session = 1

    forget_rates = np.ones(n_item) * 0.04
    repetition_rates = np.ones(n_item) * 0.80

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

    policy = Leitner(env=env)
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)

    policy = Random(env=env)
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)

    policy = ActivePragmaticOnly(env=env)
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)



if __name__ == "__main__":
    main()
