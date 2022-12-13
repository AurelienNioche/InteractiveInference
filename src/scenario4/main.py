import numpy as np
import torch

from graphic.environment import Environment

from users.fish import Fish, FishModel
from assistants.active_inference_assistant import Assistant

from plot.plot_traces import plot_traces

np.seterr(all='raise')


def main():

    seed = 123
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(seed)

    n_iteration = 200

    env = Environment(
        colors=("blue", "orange"),
        fps=30,
        hide_cursor=False)

    fish_init_position = np.array([env.size(0) * 0.5, env.size(1) * 0.5])

    fish = Fish(
        environment=env, sigma=20., jump_size=100.)
    fish.reset(init_position=fish_init_position)
    env.reset(init_shift=0, fish_init_position=fish.position)

    assistant = Assistant(
        fish_model=FishModel(environment=env,
                             jump_size=fish.jump_size,
                             sigma=fish.sigma),
        decision_rule="active_inference")

    assistant.reset()

    trace = {k: [] for k in ("assistant_belief", )}  # "user_action", "targets_positions", "assistant_action")}
    trace["assistant_belief"].append(assistant.np_belief)

    for it in range(n_iteration):
        previous_target_positions = env.target_positions.copy()
        previous_fish_position = fish.position.copy()
        fish_jump = fish.act(
            fish_position=fish.position,
            target_positions=env.target_positions,
            goal=fish.goal)
        fish.position = fish.update_fish_position(
            fish_position=previous_fish_position,
            fish_jump=fish_jump,
            window_size=env.size())
        env.update(
            new_fish_position=fish.position,
            assistant_action=None)
        assistant_action = assistant.act(
            fish_jump=fish_jump,
            previous_fish_position=previous_fish_position,
            previous_target_positions=previous_target_positions,
            new_fish_position=fish.position)
        env.update(
            assistant_action=assistant_action,
            new_fish_position=None)

        trace["assistant_belief"].append(assistant.np_belief)

    plot_traces(trace,
                save=False,
                show=True,
                run_name=None)


if __name__ == "__main__":
    main()
