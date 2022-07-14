try:
    from assistants.ai_assistant_pref_on_latent import AiAssistantPrefOnLatent
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

    run_name = "pref_on_latent"

    trace = run(
        user_model=User,
        assistant_model=AiAssistantPrefOnLatent,
        seed=1,
        user_parameters=(8.0, 0.5),
        n_step=2000,
        n_targets=10,
        target_closer_step_size=0.01,
        target_further_step_size=None,
        decision_rule='active_inference',
        decision_rule_parameters=dict(
            decay_factor=0.9,
            n_rollout=5,
            n_step_per_rollout=2))
    plot_traces(trace, show=True,
                run_name=run_name)


if __name__ == '__main__':
    main()
