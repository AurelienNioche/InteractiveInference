from . users.users import User
from . assistants.assistants import Assistant


def main():

    n_episode = 2
    max_n_step = 100

    n_targets = 10
    debug = True

    assistant = Assistant(n_targets=n_targets)
    user = User(n_targets=n_targets, debug=debug)

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
