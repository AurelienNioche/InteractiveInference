import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from baseline_policies.threshold import Threshold
from baseline_policies.conservative import Conservative
from baseline_policies.leitner import Leitner
from baseline_policies.random import Random

from active_inference.active_inference_assistant import ActiveTeacher

from environments.teaching import Teaching

# from generate_learners.generate_learners import generate_learners_parameterization


# def create_env(seed=123,
#                n_users=30,
#                n_items=200,
#                penalty_coeff=0,
#                reward_type=reward_types.MEAN_LEARNED,
#                tau=0.9,
#                n_session=6,
#                break_length=24 * 60 ** 2,
#                time_per_iter=3,
#                n_iter_session=100):
#
#     forget_rates, repetition_rates = \
#         generate_learners_parameterization(
#             n_users=n_users, n_items=n_items, seed=seed)
#
#     env = DiscontinuousTeaching(
#         tau=tau,
#         n_item=n_items,
#         n_session=n_session,
#         break_length=break_length,
#         time_per_iter=time_per_iter,
#         n_iter_per_session=n_iter_session,
#         initial_forget_rates=forget_rates,
#         repetition_rates=repetition_rates,
#         delta_coeffs=np.array([3, 20]),
#         penalty_coeff=penalty_coeff,
#         reward_type=reward_type)
#
#     return env


def run(env, policy):
    rewards = []
    actions = []

    obs = env.reset()

    with tqdm(total=env.n_iter_per_session * env.n_session) as pb:

        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            pb.update()

    final_n_learned = rewards[-1] * env.n_item
    n_view = len(np.unique(np.asarray(actions)))
    print(f"{policy.__class__.__name__.lower()} | "
          f"final reward {int(final_n_learned)} | "
          f"precision {final_n_learned / n_view:.2f}")
    return actions, rewards


def myopic(env):

    policy = Threshold(env=env)

    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)


def conservative(env):

    policy = Conservative(env=env)

    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)


def leitner(env):

    policy = Leitner(env=env, delay_factor=2, delay_min=3)

    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)


def plot(actions, rewards, env, policy):

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


def main():

    # n_items = 200
    # penalty_coeff = 0
    # reward_type = reward_types.BASE
    # tau = 0.9
    # n_session = 6
    # break_length = 24 * 60 ** 2
    # time_per_iter = 3
    # n_iter_session = 100
    #
    # forget_rates = np.ones(n_items) * 0.02
    # repetition_rates = np.ones(n_items) * 0.20

    n_items = 20
    tau = 0.9
    n_session = 6
    break_length = 10
    time_per_iter = 3
    n_iter_session = 10

    forget_rates = np.ones(n_items) * 0.005
    repetition_rates = np.ones(n_items) * 0.20

    env = Teaching(
        tau=tau,
        n_item=n_items,
        n_session=n_session,
        break_length=break_length,
        time_per_iter=time_per_iter,
        n_iter_per_session=n_iter_session,
        initial_forget_rates=forget_rates,
        repetition_rates=repetition_rates)

    policy = Conservative(env=env)
    env.reset()
    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)


    prior = torch.zeros((env.t_max, env.n_item))
    for t in range(env.t_max):
        prior[t] = torch.from_numpy(np.eye(env.n_item)[actions[t]])

    policy = ActiveTeacher(env=env, prior=prior)
    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)

    policy = Leitner(env=env, delay_factor=2, delay_min=3)
    env.reset()
    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)

    policy = Random(env=env)
    env.reset()
    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)


if __name__ == "__main__":
    main()
