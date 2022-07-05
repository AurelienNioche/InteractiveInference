from assistants.ai_assistant import AiAssistant
from users.user import User


def main():

    goal = 3

    step_size = 0.01

    learning_rate = 0.01

    n_episode = 1
    max_n_step = 100

    n_targets = 5
    debug = True

    beta = 3.0

    assistant = AiAssistant(
        step_size=step_size, n_targets=n_targets, beta=beta, learning_rate=learning_rate)
    user = User(n_targets=n_targets, beta=beta, debug=debug)

    for ep in range(n_episode):

        _ = user.reset()
        user.goal = goal
        assistant_output = assistant.reset()

        for step in range(max_n_step):

            print("Positions", assistant.x)

            user_output, _, user_done, _ = user.step(assistant_output)
            if user_done:
                break

            print("User action", user_output)

            assistant_output, _, assistant_done, _ = assistant.step(user_output)
            if assistant_done:
                break

            print()


if __name__ == '__main__':
    main()
