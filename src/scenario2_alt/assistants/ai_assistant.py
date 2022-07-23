from abc import ABC

import numpy as np
import torch
import gym
from itertools import product


class AiAssistant(gym.Env, ABC):

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

        super().__init__()

        self.n_targets = n_targets

        self.target_closer_step_size = target_closer_step_size

        if target_further_step_size is None:
            target_further_step_size = target_closer_step_size / (n_targets - 1)
        self.target_further_step_size = target_further_step_size

        self.user_model = user_model

        # Used for updating beliefs
        self.inference_max_epochs = inference_max_epochs
        self.inference_learning_rate = inference_learning_rate

        self.actions = np.arange(n_targets)  # With target to move

        self.min_x = min_x
        self.max_x = max_x

        self.starting_x = starting_x

        self.decision_rule = decision_rule
        if decision_rule_parameters is None:
            if decision_rule == "active_inference":
                decision_rule_parameters = dict(
                    decay_factor=0.9,
                    n_rollout=5,
                    n_step_per_rollout=2)
            elif decision_rule == "epsilon_rule":
                decision_rule_parameters = dict(
                    epsilon=0.1
                )
            elif decision_rule == "softmax":
                decision_rule_parameters = dict(
                    temperature=100
                )
            elif decision_rule == "random":
                decision_rule_parameters = dict()
            else:
                raise ValueError("Decision rule not recognized")

        self.decision_rule_parameters = decision_rule_parameters

        self.a = None  # Current action
        self.b = None  # Beliefs over user preferences
        self.t = None  # Iteration counter
        self.x = None  # Positions of the targets

    @property
    def belief(self):
        return torch.softmax(self.b, dim=0).detach().numpy().copy()

    @property
    def targets_position(self):
        return self.x.detach().numpy().copy()

    @property
    def action(self):
        return self.a

    def reset(self):

        self.a = None
        self.b = torch.ones(self.n_targets)
        self.x = torch.ones(self.n_targets) * self.starting_x
        observation = self.x  # np.random.choice(self.actions)
        return observation  # reward, done, info can't be included

    def step(self, action: np.ndarray):

        y = action  # sensory state is action of other user
        self.b = self.revise_belief(y=y, b=self.b, x=self.x)
        self.act()

        # print(self.a)

        observation = self.x
        done = False

        reward, info = None, None
        return observation, reward, done, info

    def revise_belief(self, y, b, x):
        """
        Update the belief based on a new observation
        """
        # Start by creating a new belief `b_prime` the previous belief `b`
        b_prime = torch.nn.Parameter(b.clone())

        b.requires_grad = True

        opt = torch.optim.Adam([b_prime, ], lr=self.inference_learning_rate)

        # Minimise free energy
        for step in range(self.inference_max_epochs):

            old_b_prime = b_prime.clone()

            opt.zero_grad()
            loss = self.free_energy_belief(b_prime=b_prime, y=y, x=x, b=b)
            loss.backward()
            opt.step()

            # if torch.isclose(old_b_prime, b_prime).all():
            #     print(f"converged at step {step}")
            #     break

        # Update internal state
        b = b_prime.detach()
        return b

    def update_target_positions(self, x, a):

        mask = np.ones(self.n_targets, dtype=bool)
        mask[a] = False

        x[mask] += self.target_further_step_size
        x[a] -= self.target_closer_step_size
        x[x > self.max_x] = self.max_x
        x[x < self.min_x] = self.min_x

        return x

    def free_energy_belief(self, b_prime, y, x, b):
        """
        Used to update `b`
        """

        p = self.user_model.complain_prob(x)

        p_y = p ** y * (1 - p) ** (1 - y)

        q_prime = torch.softmax(b_prime - b_prime.max(), dim=0)

        p_y_under_q = (p_y * q_prime).sum()
        log_p_y_under_q = p_y_under_q.log()

        log_q = torch.log_softmax(b - b.max(), dim=0)

        kl_div_q_p = torch.sum(q_prime * (q_prime.log() - log_q))
        fe = - log_p_y_under_q + kl_div_q_p
        return fe

    def act(self):

        i = getattr(self, f'_act_{self.decision_rule}')(**self.decision_rule_parameters)
        self.a = self.actions[i]
        self.update_target_positions(x=self.x, a=self.a)

    def _act_random(self):
        return np.random.choice(np.arange(self.n_targets))

    def _act_epsilon_rule(self, epsilon):
        rd = np.random.random()
        if rd < epsilon:
            i = np.random.randint(self.n_targets)
        else:
            i = np.random.choice(np.nonzero(self.b == torch.max(self.b))[0])

        return i

    def _act_softmax(self, temperature):
        p = torch.nn.functional.softmax(temperature * self.b, dim=0).numpy()
        i = np.random.choice(np.arange(self.n_targets), p=p)
        return i

    def _act_active_inference(self,
                              decay_factor,
                              n_rollout,
                              n_step_per_rollout):

        # Pick an action
        # Calculate the free energy given my target (intent) distribution, current state distribution, & sensory input
        # Do this for all actions and select the action with minimum free energy.
        kl_div = np.asarray([self.expected_free_energy_action(
            action=a,
            n_rollout=n_rollout,
            n_step_per_rollout=n_step_per_rollout,
            decay_factor=decay_factor,
        ) for a in self.actions])
        i = np.random.choice(np.nonzero(kl_div == np.min(kl_div))[0])
        return i

    def expected_free_energy_action(self, action,
                                    n_rollout,
                                    n_step_per_rollout,
                                    decay_factor):
        r"""
        KL divergence between the density function that implements preferences (:math:`Q(\mathbf{Z^*})`)
        and generative density of the same object (:math:`P(\mathbf{X}, \mathbf{Z})`)

        .. math::
            \begin{align}
                &-\mathbb {E} _{\mathbf {Z} \sim Q} \left[\log \frac {P(\mathbf {X} ,\mathbf {Z} )}{Q(\mathbf {Z} )} \right] \\
                =& D_{\mathrm {KL} }(Q(\mathbf {Z})  \parallel P(\mathbf{X}, \mathbf{Z}))
            \end{align}
        """

        if n_step_per_rollout > 1:
            efe = []
            for _ in range(n_rollout):
                efe_rollout = 0

                x = self.x.clone()
                b = self.b.clone()

                action_plan = np.zeros(n_step_per_rollout, dtype=int)
                action_plan[0] = action
                action_plan[1:] = np.random.choice(self.actions, size=n_step_per_rollout-1)

                for step, action in enumerate(action_plan):

                    self.update_target_positions(x=x, a=action)

                    p = self.user_model.complain_prob(x)

                    q = torch.softmax(b - b.max(), dim=0)

                    p_y = 1 - p  # We don't want the user to complain
                    p_y_under_q = (p_y * q).sum()

                    extrinsic_value = p_y_under_q

                    y = (torch.rand(1) < p_y_under_q).long()
                    b_new = self.revise_belief(y=y, b=b, x=x)
                    q_new = torch.softmax(b_new - b_new.max(), dim=0)

                    epistemic_value = torch.sum(q_new * (q_new.log() - q.log()))

                    efe_step = - extrinsic_value - epistemic_value

                    efe_rollout += decay_factor**step * efe_step

                    b = b_new

                efe.append(efe_rollout.item())

            return np.mean(efe)

        else:

            x = self.x.clone()
            b = self.b.clone()

            self.update_target_positions(x=x, a=action)

            p = self.user_model.complain_prob(x)

            q = torch.softmax(b - b.max(), dim=0)

            p_y = 1 - p  # We don't want the user to complain
            p_y_under_q = (p_y * q).sum()

            extrinsic_value = p_y_under_q

            efe_step = - extrinsic_value

            for y in torch.arange(2):

                p_y = p ** y * (1 - p) ** (1 - y)
                p_y_under_q = (p_y * q).sum()

                b_new = self.revise_belief(y=y, b=b, x=x)
                q_new = torch.softmax(b_new - b_new.max(), dim=0)

                epistemic_value = torch.sum(q_new * (q_new.log() - q.log()))

                efe_step += - p_y_under_q*epistemic_value

            return efe_step.item()
