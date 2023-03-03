import numpy as np

from run.run import run
from plot.plot import plot

from active_inference.active_inference_epistemic_only import ActiveEpistemic

from environments.teaching import Teaching


def main():

    n_item = 20
    tau = 0.9
    n_session = 3
    break_length = 24 * 60 ** 2     # 24 * 60 ** 2
    time_per_iter = 3               # 3
    n_iter_session = 10              # 20

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

    print(f"N item {n_item} | T_max {env.t_max}")

    # policy = ConservativeNotOmniscient(
    #     env=env,
    #     bounds=np.asarray([[0.01, 0.01], [0.0, 0.5]]),
    #     grid_methods=('lin', 'lin'),
    #     grid_size=100)
    # env.reset()
    # traces = run(env=env, policy=policy)
    # plot(traces, env=env, policy=policy)

    policy = ActiveEpistemic(
        env=env,
        bounds=np.asarray([[0.01, 0.01], [0.0, 0.5]]),
        grid_methods=('geo', 'lin'),
        grid_size=50)
    env.reset()
    traces = run(env=env, policy=policy)
    plot(traces, env=env, policy=policy)

    # policy = Conservative(env=env)
    # env.reset()
    # traces = run(env=env, policy=policy)
    # plot(traces, env=env, policy=policy)

    # policy = ActivePragmaticOnly(
    #     env=env,
    #     param_kwargs={"learning_rate": 0.5, "max_epochs": 100, "n_sample": 5})
    # env.reset()
    # traces = run(env=env, policy=policy)
    # plot(traces, env=env, policy=policy)

    # prior = torch.zeros((env.t_max, env.n_item))
    # for t in range(env.t_max):
    #     prior[t] = torch.from_numpy(np.eye(env.n_item)[actions[t]])
    #
    # policy = ActiveTeacher(env=env, prior=prior)
    # actions, rewards = run(env=env, policy=policy)
    # plot(actions=actions, rewards=rewards, env=env, policy=policy)

    # policy = Leitner(env=env, delay_factor=2, delay_min=3)
    # env.reset()
    # traces = run(env=env, policy=policy)
    # plot(traces, env=env, policy=policy)

    # policy = Random(env=env)
    # env.reset()
    # actions, rewards = run(env=env, policy=policy)
    # plot(actions=actions, rewards=rewards, env=env, policy=policy)


if __name__ == "__main__":
    main()