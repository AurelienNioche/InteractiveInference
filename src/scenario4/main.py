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

    fish_init_position = np.array([env.size(0)* 1.0, env.size(1) * 0.5])

    fish = Fish(
        environment=env, sigma=10., jump_size=3.)
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
        new_fish_position = fish.update_fish_position(
            fish_position=previous_fish_position,
            fish_jump=fish_jump,
            window_size=env.size())
        env.update(
            new_fish_position=new_fish_position,
            assistant_action=None)
        assistant_action = assistant.act(
            fish_jump=fish_jump,
            previous_fish_position=previous_fish_position,
            previous_target_positions=previous_target_positions,
            new_fish_position=new_fish_position)
        env.update(
            assistant_action=assistant_action,
            new_fish_position=None)

        trace["assistant_belief"].append(assistant.np_belief)

    # user = User(**user_kwargs, goal=0)
    # user.reset()
    # user_model = UserModel(**user_kwargs)
    # user_model.reset()
    #
    # assistant = Assistant(user_model=user_model, n_target=n_target,
    #                       window=display.window,
    #                       # decision_rule="random")
    #                       decision_rule="active_inference")
    # assistant.reset()
    #
    # target_positions = assistant.target_positions
    # user_action = np.zeros(2)
    #
    # trace = {k: [] for k in ("assistant_belief", "user_action", "targets_positions", "assistant_action")}
    #

    #
    #     trace["targets_positions"].append(display.target_positions)
    #     trace["assistant_belief"].append(assistant.belief)
    #
    #     display.update(user_action=user_action, assistant_action=target_positions)
    #     user_action = user.act(target_positions)
    #
    #     target_positions = assistant.act(user_action=user_action)
    #
    #     trace["user_action"].append(user.action)
    #     trace["assistant_action"].append(assistant.action)  # This is not positions, but angles
    #
    # trace["preferred_target"] = user.goal
    #
    plot_traces(trace,
                save=False,
                show=True,
                run_name=None)


if __name__ == "__main__":
    main()
