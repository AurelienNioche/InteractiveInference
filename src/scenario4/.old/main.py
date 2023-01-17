import numpy as np
import torch

from graphic.Display import Environment

from users.fish import Fish, FishModel
from assistants.active_inference_assistant import Assistant

from plot.plot_traces import plot_traces

np.seterr(all='raise')


def main():

    seed = 123
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    n_iteration = 200

    env = Environment(
        colors=("blue", "orange"),
        fps=30,
        hide_cursor=False)
    env.reset(fish_init_position=torch.tensor([0, env.size(1) * 0.5]),
              init_shift=torch.zeros(1))

    fish = Fish(
        environment=env, sigma=10., movement_amplitude=3.)

    assistant = Assistant(
        fish_model=FishModel(environment=env,
                             movement_amplitude=fish.jump_size,
                             sigma=fish.sigma),
        decision_rule="active_inference")

    assistant.reset()

    trace = {k: [] for k in ("assistant_belief", )}  # "user_action", "targets_positions", "assistant_action")}
    trace["assistant_belief"].append(assistant.np_belief)

    for it in range(n_iteration):
        previous_target_positions = env.target_positions.clone()
        previous_fish_position = env.fish_position.clone()
        fish_jump = fish.act(
            fish_position=env.fish_position,
            target_positions=env.target_positions,
            goal=fish.goal)
        env.update(
            user_action=fish_jump,
            assistant_action=None)
        assistant_action = assistant.act(
            fish_jump=fish_jump,
            previous_fish_position=previous_fish_position,
            previous_target_positions=previous_target_positions)
        env.update(
            user_action=None,
            assistant_action=assistant_action)

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
