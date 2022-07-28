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
    n_steps = 200

    decision_rule_kwargs = dict(
        decision_rule='active_inference',
        decision_rule_parameters=dict(
            efe='efe_on_latent',
            decay_factor=0.9,
            n_rollout=1,
            n_step_per_rollout=1)
    )

    # decision_rule_kwargs = dict(
    #     decision_rule='epsilon_rule',
    #     decision_rule_parameters=dict(
    #         epsilon=0.1)
    # )

    # decision_rule_kwargs = dict(
    #     decision_rule="random"
    # )

    run_name = f"{decision_rule_kwargs['decision_rule']}" \
               f"_{n_targets}targets_{n_steps}steps"

    if decision_rule_kwargs['decision_rule'] == "active_inference":
        n_steps_per_rollout = decision_rule_kwargs['decision_rule_parameters']['n_step_per_rollout']
        if n_steps_per_rollout > 1:
            n_rollout = decision_rule_kwargs['decision_rule_parameters']['n_rollout']
            run_name += f"_{n_steps_per_rollout}steps_per_rollout_{n_rollout}rollout"

        efe = decision_rule_kwargs['decision_rule_parameters']['efe']
        if efe != "efe_on_obs":
            run_name += f"_{efe}"

    trace = run(
        user_model=User,
        assistant_model=AiAssistant,
        seed=1,
        user_parameters=(8.0, 0.5),
        n_step=n_steps,
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
