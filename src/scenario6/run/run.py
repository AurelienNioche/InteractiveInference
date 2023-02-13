from tqdm import tqdm
import numpy as np
from types import SimpleNamespace


def run(env, policy):

    rewards = []
    actions = []
    inferred_parameters = []

    obs = env.reset()

    with tqdm(total=env.n_iter_per_session * env.n_session) as pb:

        done = False
        while not done:
            action = policy.act(obs)
            if hasattr(policy, "psy"):
                inferred_parameters.append(policy.psy.est_param.copy())
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            pb.update()

    final_n_learned = rewards[-1] * env.n_item
    n_view = len(np.unique(np.asarray(actions)))
    print(f"{policy.__class__.__name__.lower()} | "
          f"final reward {int(final_n_learned)} | "
          f"precision {final_n_learned / n_view:.2f}")

    return SimpleNamespace(**{"actions": actions, "rewards": rewards, "inferred_parameters": inferred_parameters})
