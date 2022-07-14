from abc import ABC

import numpy as np
import torch
import gym


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
            else:
                raise ValueError("Decision rule not recognized")

        self.decision_rule_parameters = decision_rule_parameters

        self.a = None  # Current action
        self.b = None  # Beliefs over user preferences
        self.t = None  # Iteration counter
        self.x = None  # Positions of the targets

    @property
    def belief(self):
        return self.variational_density(self.b).detach().numpy()

    @property
    def action(self):
        return self.a

    def reset(self):

        self.a = None
        self.b = torch.ones(self.n_targets)
        self.x = torch.ones(self.n_targets) * self.starting_x
        observation = self.x.detach().numpy().copy()  # np.random.choice(self.actions)
        return observation  # reward, done, info can't be included

    def step(self, action: np.ndarray):

        y = action  # sensory state is action of other user
        self.b = self.revise_belief(y=y, b=self.b, x=self.x)
        self.act()

        # print(self.a)

        observation = self.x.detach().numpy().copy()
        done = False

        reward, info = None, None
        return observation, reward, done, info

    @staticmethod
    def kl_div(a, b):
        """
        Kullback-Leibler divergence between densities a and b.
        """
        # If access to distributions "registered" for KL div, then:
        # torch.distributions.kl.kl_divergence(p, q)
        # Otherwise:
        # torch.nn.functional.kl_div(input=torch.log(b), target=a)
        # Note the inversion between the arguments order,
        # and the fact that you need to give the first in the log-space
        return torch.sum(a * (torch.log(a) - torch.log(b)))

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

            if torch.isclose(old_b_prime, b_prime).all():
                break

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

    def generative_density_belief(self, y, x, b):
        r"""
        Compute the generative density.

        .. math::
           P(\mathbf{X}, \mathbf{Z}) = P(\mathbf{X} \mid \mathbf{Z}) P(\mathbf{Z})
        """

        p_preferred = self.variational_density(b)
        cond_p = self.user_model.conditional_probability_action(x)

        if not y:
            cond_p = 1 - cond_p

        joint_p = cond_p * p_preferred
        return joint_p

    def free_energy_belief(self, b_prime, y, x, b):
        r"""
        KL divergence between variational density (:math:`Q(\mathbf{Z})`)
        and generative density (:math:`P(\mathbf{X}, \mathbf{Z})`)

        .. math::
            \begin{align}
                &-\mathbb {E} _{\mathbf {Z} \sim Q} \left[\log \frac {P(\mathbf {X} ,\mathbf {Z} )}{Q(\mathbf {Z} )} \right] \\
                =& D_{\mathrm {KL} }(Q(\mathbf {Z})  \parallel P(\mathbf{X}, \mathbf{Z}))
            \end{align}

        Used to update `b`
        """
        # KL(target, input)
        return self.kl_div(
            a=self.variational_density(b_prime),
            b=self.generative_density_belief(y=y, x=x, b=b))

    @staticmethod
    def variational_density(b):
        r"""
        :math:`Q(\mathbf{Z})`

        Agent's beliefs.
        """
        # Softmax function. The shift by b.max() is for numerical stability
        b_scaled = b - b.max()
        return torch.nn.functional.softmax(b_scaled, dim=0)

    def act(self):

        i = getattr(self, f'_act_{self.decision_rule}')(**self.decision_rule_parameters)
        self.a = self.actions[i]
        self.update_target_positions(x=self.x, a=self.a)

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
            a=a,
            decay_factor=decay_factor,
            n_rollout=n_rollout,
            n_step_per_rollout=n_step_per_rollout,
        ).item() for a in self.actions])
        i = np.random.choice(np.nonzero(kl_div == np.min(kl_div))[0])
        return i

    @property
    def target_dist(self):
        """
        The target distribution ("preferences") is to be certain that the user would not complain
        """
        return torch.tensor([1, 0]) + 1e-8

    def generative_density_action(self, x, b):

        joint_complains = self.generative_density_belief(y=1, x=x, b=b)
        marginalized_complains = joint_complains.sum()

        return torch.tensor([1 - marginalized_complains, marginalized_complains])

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
                gd = self.generative_density_action(x=x, b=b)
                kl = self.kl_div(a=self.target_dist,
                                 b=gd)

                kl_rollout.append(decay_factor**step * kl)

                if step < (n_step_per_rollout - 1):

                    y = np.random.random() < gd[1]
                    b = self.revise_belief(y=y, b=self.b, x=self.x)

        return np.sum(kl_rollout)
