import numpy as np
import gym
from gym import spaces

from agents.users import User
from agents.assistants.assistants_act_inf import AiAssistant


def main():

    n_episode = 2
    max_n_step = 100

    n_targets = 10
    debug = True

    beta = 1.0

    assistant = AiAssistant(n_targets=n_targets, beta=beta)
    user = User(n_targets=n_targets, beta=beta, debug=debug)

    for ep in range(n_episode):

        _ = user.reset()
        assistant_output = assistant.reset()

        for step in range(max_n_step):

            user_output, _, user_done, _ = user.step(assistant_output)
            if user_done:
                break

            assistant_output, _, assistant_done, _ = assistant.step(user_output)
            if assistant_done:
                break


if __name__ == '__main__':
    main()
