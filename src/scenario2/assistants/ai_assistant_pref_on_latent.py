from abc import ABC

import numpy as np
import torch

from . ai_assistant import AiAssistant


class AiAssistantPrefOnLatent(AiAssistant):

    """
    x: position of the targets
    psi: latent state; preferences of the user for each target
    actions: moving one target closer, and moving away all the other targets
    b_star: preferences of the assistant; preferences relate to the distance to the preferred target
    """

    def __init__(self, n_targets,
                 user_model,
                 starting_x=0.5,
                 target_closer_step_size=0.1,
                 target_further_step_size=None,
                 min_x=0.0,
                 max_x=1.0,
                 inference_learning_rate=0.1,
                 inference_max_epochs=500,
                 decision_rule='active_inference',
                 decision_rule_parameters=None):

        super().__init__(
            n_targets=n_targets,
            user_model=user_model,
            starting_x=starting_x,
            target_closer_step_size=target_closer_step_size,
            target_further_step_size=target_further_step_size,
            min_x=min_x,
            max_x=max_x,
            inference_learning_rate=inference_learning_rate,
            inference_max_epochs=inference_max_epochs,
            decision_rule=decision_rule,
            decision_rule_parameters=decision_rule_parameters
        )

    def kl_div_action(self, x, b):

        scale_q = 0.05
        n_sample = 100
        q_dist = torch.distributions.HalfNormal(scale_q)
        q_samples = q_dist.sample((100,))
        log_q = q_dist.log_prob(q_samples)

        scale_target = 0.01
        dist_target = [torch.distributions.Normal(x[t], scale_target) for t in range(self.n_targets)]
        log_p = torch.zeros((self.n_targets, n_sample))
        for t in range(self.n_targets):
            log_p[t] = dist_target[t].log_prob(q_samples) * b[t]
        log_p = log_p.sum(dim=0)
        return torch.mean(log_q - log_p).item()

    def expected_free_energy_action(self, a,
                                    decay_factor,
                                    n_rollout,
                                    n_step_per_rollout):
        r"""
        KL divergence between the density function that implements preferences (:math:`Q(\mathbf{Z^*})`)
        and generative density of the same object (:math:`P(\mathbf{X}, \mathbf{Z})`)

        .. math::
            \begin{align}
                &-\mathbb {E} _{\mathbf {Z} \sim Q} \left[\log \frac {P(\mathbf {X} ,\mathbf {Z} )}{Q(\mathbf {Z} )} \right] \\
                =& D_{\mathrm {KL} }(Q(\mathbf {Z})  \parallel P(\mathbf{X}, \mathbf{Z}))
            \end{align}
        """

        kl_rollout = []

        for _ in range(n_rollout):

            x = self.x.clone()
            b = self.b.clone()

            for step in range(n_step_per_rollout):
                if step == 0:
                    action = a
                else:
                    # Choose randomly a new action
                    action = np.random.choice(self.actions)

                self.update_target_positions(x=x, a=action)
                kl = self.kl_div_action(x=x, b=b)

                kl_rollout.append(decay_factor ** step * kl)

                if step < (n_step_per_rollout - 1):

                    gd = self.generative_density_action(x=x, b=b)
                    y = np.random.random() < gd[1]
                    b = self.revise_belief(y=y, b=self.b, x=self.x)

        return np.sum(kl_rollout)

    # def b_star(self, x, scale=0.05):
    #     """
    #     b_star is about the distance to the desired target
    #     The closer, the better
    #     :return:
    #     """
    #     return torch.distributions.HalfNormal(scale).log_prob(x+0.00001)

    # def free_energy_action(self, b_star, b, x, a):
    #     """
    #     KL divergence between variational density and generative density for a fixed
    #     sensory state s.
    #     """
    #     p_preferred = self.variational_density(b)
    #
    #     x = self.take_action(x=x.clone(), a=a)
    #
    #     # dist = torch.zeros(len(self.dist_space))
    #     # for i, x_ in enumerate(self.dist_space):
    #     #     for j in range(self.n_targets):
    #     #         p_j, x_j = p[j], x_temp[j]
    #     #         if np.isclose(x_j, x_):
    #     #             dist[i] += p_j
    #
    #     kl = 0
    #     for i in range(self.n_targets):
    #         p_i, x_i = p_preferred[i], x[i]
    #         a = p_i
    #         log_b = b_star(x_i)
    #         kl += a * (torch.log(a) - log_b)
    #
    #     # We use the reverse KL divergence here
    #     # return self.KL(b_star,
    #     #                dist)
    #     return kl
