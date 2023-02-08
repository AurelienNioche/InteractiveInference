import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from baseline_policies.threshold import Threshold
from baseline_policies.conservative import Conservative
from baseline_policies.leitner import Leitner

#  from environments.continuous_teaching import ContinuousTeaching
from environments.discontinuous_teaching import DiscontinuousTeaching
from environments import reward_types

from generate_learners.generate_learners import generate_learners_parameterization


def create_env(seed=123,
               n_users=30,
               n_items=200,
               penalty_coeff=0,
               reward_type=reward_types.MEAN_LEARNED,
               tau=0.9,
               n_session=6,
               break_length=24 * 60 ** 2,
               time_per_iter=3,
               n_iter_session=100):

    forget_rates, repetition_rates = \
        generate_learners_parameterization(
            n_users=n_users, n_items=n_items, seed=seed)

    env = DiscontinuousTeaching(
        tau=tau,
        n_item=n_items,
        n_session=n_session,
        break_length=break_length,
        time_per_iter=time_per_iter,
        n_iter_per_session=n_iter_session,
        initial_forget_rates=forget_rates,
        initial_repetition_rates=repetition_rates,
        delta_coeffs=np.array([3, 20]),
        penalty_coeff=penalty_coeff,
        reward_type=reward_type)

    return env


def run(env, policy):
    rewards = []
    actions = []

    obs = env.reset()

    with tqdm(total=env.n_iter_per_session * env.n_session) as pb:
        while True:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            if done:
                # Simulate exam
                obs, reward, done, _ = env.step(None)
                rewards.append(reward)
                break

            pb.update()

    final_n_learned = reward * env.n_item
    n_view = len(np.unique(np.asarray(actions)))
    print(f"{policy.__class__.__name__.lower()} | "
          f"final reward {int(final_n_learned)} | "
          f"precision {final_n_learned / n_view:.2f}")
    return actions, rewards


def test_myopic(seed=123):

    env = create_env(seed=seed)
    policy = Threshold(env=env)

    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)


def test_conservative(seed=123):

    env = create_env(seed=seed)
    policy = Conservative(env=env)

    actions, rewards = run(env=env, policy=policy)
    plot(actions=actions, rewards=rewards, env=env, policy=policy)


def test_leitner(seed=123):

    env = create_env(seed=seed)
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

    seed = 0
    test_leitner(seed=seed)
    test_myopic(seed=seed)
    test_conservative(seed=seed)


if __name__ == "__main__":
    main()
