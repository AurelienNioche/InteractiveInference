try:
    from assistants.ai_assistant import AiAssistant
    from users.user import User
    from run.run import run
    from plot.plot import plot_traces

# For documentation
except ModuleNotFoundError:
    # noinspection PyUnresolvedReferences
    from scenario2.assistants.ai_assistant import AiAssistant
    # noinspection PyUnresolvedReferences
    from scenario2.users.user import User
    # noinspection PyUnresolvedReferences
    from scenario2.plot.plot import plot_traces
    # noinspection PyUnresolvedReferences
    from scenario2.run.run import run


def main():

    # decision_rule_kwargs = dict(
    #     decision_rule='softmax',
    #     decision_rule_parameters=dict(
    #         temperature=100.0)
    # )

    n_targets = 10

    decision_rule_kwargs = dict(
        decision_rule='active_inference',
        decision_rule_parameters=dict(
            decay_factor=0.9,
            n_rollout=None,
            n_step_per_rollout=1)
    )

    # decision_rule_kwargs = dict(
    #     decision_rule='epsilon_rule',
    #     decision_rule_parameters=dict(
    #         epsilon=0.2)
    # )

    # decision_rule_kwargs = dict(
    #     decision_rule="random"
    # )

    run_name = f"{decision_rule_kwargs['decision_rule']}_{n_targets}targets"

    trace = run(
        user_model=User,
        assistant_model=AiAssistant,
        seed=1,
        user_parameters=(8.0, 0.5),
        n_step=1000,
        n_targets=n_targets,
        inference_max_epochs=500,
        inference_learning_rate=0.1,
        target_closer_step_size=0.01,
        target_further_step_size=None,
        **decision_rule_kwargs)

    plot_traces(trace, show=True,
                run_name=run_name)


if __name__ == '__main__':
    main()
