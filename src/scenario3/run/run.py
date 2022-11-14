from typing import Iterable
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
    user_goal: Iterable[float] = (0.8, 0.1),
    seed: int = 1234,
    inference_max_epochs: int = 500,
    inference_learning_rate: float = 0.01,
    n_step: int = 200,
    user_parameters: [tuple, list, float, None] = None,
    decision_rule: str = "active_inference",
    decision_rule_parameters: [dict, None] = None,
) -> dict:

    np.random.seed(seed)
    torch.manual_seed(seed)

    trace = {k: [] for k in ("assistant_belief", "user_action",
                             "targets_position", "assistant_action")}

    user_goal = torch.tensor(user_goal)

    user = user_model(
        goal=user_goal,
        parameters=user_parameters)

    assistant = assistant_model(
        user_model=user,
        inference_learning_rate=inference_learning_rate,
        inference_max_epochs=inference_max_epochs,
        decision_rule=decision_rule,
        decision_rule_parameters=decision_rule_parameters)

    _ = user.reset()
    assistant_output = assistant.reset()

    for _ in tqdm(range(n_step)):

        trace["assistant_action"].append(assistant.action)
        trace["assistant_belief"].append(assistant.belief)

        user_output, _, user_done, _ = user.act(assistant_output)
        if user_done:
            break

        trace["user_action"].append(user.action)

        assistant_output, _, assistant_done, _ = assistant.act(user_output)
        if assistant_done:
            break

    trace["user_goal"] = user.goal

    return trace
