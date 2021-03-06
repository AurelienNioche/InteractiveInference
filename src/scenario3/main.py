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

    decision_rule_kwargs = dict(
        decision_rule='active_inference',
        decision_rule_parameters=dict(
            action_max_epochs=50,
            action_learning_rate=0.1,
            n_sample=10)
    )

    # decision_rule_kwargs = dict(
    #     decision_rule='epsilon_rule',
    #     decision_rule_parameters=dict(
    #         epsilon=0.2))

    # decision_rule_kwargs = dict(
    #     decision_rule='random',
    # )

    run_name = f"{decision_rule_kwargs['decision_rule']}"

    trace = run(
        user_model=User,
        user_goal=[0.4, 0.8],
        assistant_model=AiAssistant,
        seed=2,
        inference_max_epochs=50,
        user_parameters=(100.0, 0.05),
        inference_learning_rate=0.1,
        n_step=300,
        **decision_rule_kwargs)

    plot_traces(trace, show=True,
                run_name=run_name)


if __name__ == '__main__':
    main()
