import numpy as np
import torch
import multiprocessing

from graphic.display import Display

from users.fish import Fish, FishModel
from assistants.active_inference_assistant import Assistant
from environment.environment import Environment

from plot.plot_traces import plot_traces

np.seterr(all='raise')


def run(queue, n_target, screen_size, n_iteration=200):

    fish_position = 0.5 * np.asarray(screen_size[:])
    init_shift = 0

    env = Environment(n_target=n_target)
    fish = Fish(sigma=20., jump_size=100.)
    assistant = Assistant(
        fish_model=FishModel(n_target=n_target,
                             jump_size=fish.jump_size,
                             sigma=fish.sigma),
        decision_rule="active_inference")

    target_positions = env.update_target_positions(
        shift=init_shift,
        screen_size=screen_size)

    # trace = {k: [] for k in ("assistant_belief", )}  # "user_action", "targets_positions", "assistant_action")}
    # trace["assistant_belief"].append(assistant.np_belief)

    for iteration in range(n_iteration):

        previous_target_positions = target_positions.copy()
        previous_fish_position = fish_position.copy()

        fish_jump = fish.act(
            fish_position=fish_position,
            target_positions=target_positions,
            goal=fish.goal,
            screen_size=screen_size)
        fish_position = fish.update_fish_position(
            fish_position=previous_fish_position,
            fish_jump=fish_jump,
            screen_size=screen_size)

        queue.put(dict(fish_position=fish_position, target_positions=target_positions))

        target_shift = assistant.act(
            fish_jump=fish_jump,
            fish_position=fish_position,
            previous_fish_position=previous_fish_position,
            previous_target_positions=previous_target_positions,
            update_target_positions=env.update_target_positions,
            screen_size=screen_size)

        target_positions = env.update_target_positions(shift=target_shift, screen_size=screen_size)

        queue.put(dict(fish_position=fish_position, target_positions=target_positions))

        # trace["assistant_belief"].append(assistant.np_belief)


def main():

    seed = 123
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(seed)

    colors = ("blue", "orange")

    disp = Display(
        colors=colors,
        fps=30,
        hide_cursor=False)

    screen_size = multiprocessing.Array('i', disp.size().copy())

    queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=run,
                                kwargs=dict(
                                    screen_size=screen_size,
                                    queue=queue, n_target=len(colors)))
    p.start()

    while True:

        try:
            if not queue.empty():
                from_queue = queue.get()
                disp.update(**from_queue)

            disp.graphic_update()
            screen_size[:] = disp.size()
        except SystemExit:
            p.terminate()
            break

    # plot_traces(trace,
    #             save=False,
    #             show=True,
    #             run_name=None)


if __name__ == "__main__":
    main()
