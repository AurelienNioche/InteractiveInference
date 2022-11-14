import numpy as np
import torch
from tqdm import tqdm

try:
    from assistants.ai_assistant import AiAssistant
    from users.user import User
    from plot.plot import plot_traces

# For documentation
except ModuleNotFoundError:
    # noinspection PyUnresolvedReferences
    from scenario2.assistants.ai_assistant import AiAssistant
    # noinspection PyUnresolvedReferences
    from scenario2.users.user import User
    # noinspection PyUnresolvedReferences
    from scenario2.plot.plot import plot_traces


def run(
    user_model,
    assistant_model,
    seed: int = 1234,
    goal: int = 3,
    n_targets: int = 5,
    target_closer_step_size: float = 0.01,
    target_further_step_size: [float, None] = None,
    inference_max_epochs: int = 500,
    inference_learning_rate: float = 0.01,
    n_step: int = 200,
    user_parameters: [tuple, list, float, None] = None,
    decision_rule: str = "active_inference",
    decision_rule_parameters: [dict, None] = None,
) -> dict:

    assert goal < n_targets

    np.random.seed(seed)
    torch.manual_seed(seed)

    trace = {k: [] for k in ("assistant_belief", "user_action",
                             "targets_position", "assistant_action")}

    user = user_model(
        goal=goal,
        n_targets=n_targets,
        parameters=user_parameters)

    assistant = assistant_model(
        user_model=user,
        target_closer_step_size=target_closer_step_size,
        target_further_step_size=target_further_step_size,
        n_targets=n_targets,
        inference_learning_rate=inference_learning_rate,
        inference_max_epochs=inference_max_epochs,
        decision_rule=decision_rule,
        decision_rule_parameters=decision_rule_parameters)

    _ = user.reset()
    assistant_output = assistant.reset()

    for _ in tqdm(range(n_step)):

        trace["targets_position"].append(assistant.targets_position)
        trace["assistant_belief"].append(assistant.belief)

        user_output, _, user_done, _ = user.act(assistant_output)
        if user_done:
            break

        assistant_output, _, assistant_done, _ = assistant.act(user_output)
        if assistant_done:
            break

        trace["user_action"].append(user.action)
        trace["assistant_action"].append(assistant.action)

    trace["preferred_target"] = user.goal

    return trace
