import numpy as np

from graphic.display import Display

from users.fish import Fish, FishModel
from assistants.active_inference_assistant import Assistant

from plot.plot_traces import plot_traces

np.seterr(all='raise')


def main():

    colors = ("orange", "blue")

    display_kwargs = dict(
        fps=30,
        colors=colors,
        control_using_mouse=True,
        control_per_artificial_user=True,
        hide_cursor=False,
    )

    n_target = len(display_kwargs["colors"])

    user_kwargs = dict(
        sigma=20.0,
        n_target=n_target
    )

    n_iteration = 200

    display = Display(**display_kwargs)
    display.reset()
    user = User(**user_kwargs, goal=0)
    user.reset()
    user_model = UserModel(**user_kwargs)
    user_model.reset()

    assistant = Assistant(user_model=user_model, n_target=n_target,
                          window=display.window,
                          # decision_rule="random")
                          decision_rule="active_inference")
    assistant.reset()

    target_positions = assistant.target_positions
    user_action = np.zeros(2)

    trace = {k: [] for k in ("assistant_belief", "user_action", "targets_positions", "assistant_action")}

    for it in range(n_iteration):

        trace["targets_positions"].append(display.target_positions)
        trace["assistant_belief"].append(assistant.belief)

        display.update(user_action=user_action, assistant_action=target_positions)
        user_action = user.act(target_positions)

        target_positions = assistant.act(user_action=user_action)

        trace["user_action"].append(user.action)
        trace["assistant_action"].append(assistant.action)  # This is not positions, but angles

    trace["preferred_target"] = user.goal

    plot_traces(trace,
                save=False,
                show=True,
                run_name=None)


if __name__ == "__main__":
    main()
