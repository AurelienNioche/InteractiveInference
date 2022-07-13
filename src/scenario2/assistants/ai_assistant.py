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

    def __init__(self, n_targets, learning_rate,
                 user_model,
                 starting_x=0.5,
                 step_size=0.1,
                 step_size_other=None,
                 min_x=0.0,
                 max_x=1.0,
                 max_n_epochs=500,
                 decay_factor=0.9,
                 n_rollout=5,
                 n_step_per_rollout=2,
                 decision_rule='active_inference',
                 decision_rule_parameters=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.n_targets = n_targets

        self.step_size = step_size

        if step_size_other is None:
            step_size_other = step_size / (n_targets - 1)
        self.step_size_other = step_size_other

        self.user_model = user_model

        # Used for updating beliefs
        self.max_n_epochs = max_n_epochs
        self.learning_rate = learning_rate

        self.actions = np.arange(n_targets)  # With target to move

        self.min_x = min_x
        self.max_x = max_x

        self.starting_x = starting_x

        self.decision_rule = decision_rule
        if decision_rule_parameters is None:
            decision_rule_parameters = {}
        self.decision_rule_parameters = decision_rule_parameters

        self.decay_factor = decay_factor

        self.n_rollout = n_rollout
        self.n_step_per_rollout = n_step_per_rollout

        self.a = None  # Current action
        self.b = None  # Beliefs over user preferences
        self.t = None  # Iteration counter
        self.x = None  # Positions of the targets

    @property
    def belief(self):
        return self.variational_density(self.b).detach().numpy()

    def reset(self):

        self.a = None
        self.b = torch.ones(self.n_targets)
        self.x = torch.ones(self.n_targets) * self.starting_x
        observation = self.x.detach().numpy().copy()  # np.random.choice(self.actions)
        return observation  # reward, done, info can't be included

    @property
    def target_dist(self):
        """
        The target distribution ("preferences") is to be certain that the user would not complain
        """
        return torch.tensor([1, 0]) + 1e-8

    def revise_belief(self, y, b, x):

        # Update my internal state.
        # Start by creating a new belief b_prime, from my belief for the previous belief b
        b_prime = torch.nn.Parameter(b.clone())

        b.requires_grad = True

        opt = torch.optim.Adam([b_prime, ], lr=self.learning_rate)
        # Now minimise free energy
        for step in range(self.max_n_epochs):

            old_b_prime = b_prime.clone()

            opt.zero_grad()
            loss = self.free_energy_beliefs(b_prime=b_prime, y=y, x=x, b=b)
            loss.backward()
            opt.step()

            if torch.isclose(old_b_prime, b_prime).all():
                break

        # Update internal state
        b = b_prime.detach()
        return b

    def act(self):

        i = getattr(self, f'_act_{self.decision_rule}')(**self.decision_rule_parameters)
        self.a = self.actions[i]
        self.update_target_positions(x=self.x, a=self.a)

    def _act_epsilon_rule(self, epsilon=0.1):
        rd = np.random.random()
        if rd < epsilon:
            i = np.random.randint(self.n_targets)
        else:
            i = np.random.choice(np.nonzero(self.b == torch.max(self.b))[0])

        return i

    def _act_softmax(self, temperature=100):
        p = torch.nn.functional.softmax(temperature*self.b, dim=0).numpy()
        i = np.random.choice(np.arange(self.n_targets), p=p)
        return i

    def _act_active_inference(self):

        # Pick an action
        # Calculate the free energy given my target (intent) distribution, current state distribution, & sensory input
        # Do this for all actions and select the action with minimum free energy.
        kl_div = np.asarray([self.expected_free_energy_action(a=a).item() for a in self.actions])
        i = np.random.choice(np.nonzero(kl_div == np.min(kl_div))[0])
        return i

    def step(self, action: np.ndarray):

        y = action  # sensory state is action of other user
        self.b = self.revise_belief(y=y, b=self.b, x=self.x)
        self.act()

        # print(self.a)

        observation = self.x.detach().numpy().copy()
        done = False

        reward, info = None, None
        return observation, reward, done, info

    def update_target_positions(self, x, a):

        mask = np.ones(self.n_targets, dtype=bool)
        mask[a] = False

        x[mask] += self.step_size_other
        x[a] -= self.step_size
        x[x > self.max_x] = self.max_x
        x[x < self.min_x] = self.min_x

        return x

    def generative_density(self, y, x, b):
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

    @staticmethod
    def variational_density(b):
        """
        :math:`Q(\mathbf{Z})`

        Agent's beliefs.
        """
        # Softmax function. The shift by b.max() is for numerical stability
        b_scaled = b - b.max()
        return torch.nn.functional.softmax(b_scaled, dim=0)

    def expected_free_energy_action(self, a):
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

        for _ in range(self.n_rollout):

            x = self.x.clone()
            b = self.b.clone()

            for step in range(self.n_step_per_rollout):
                if step == 0:
                    action = a
                else:
                    # Choose randomly a new action
                    action = np.random.choice(self.actions)

                self.update_target_positions(x=x, a=action)

                joint_complains = self.generative_density(y=1, x=x, b=b)
                marginalized_complains = joint_complains.sum()

                gd = torch.tensor([1 - marginalized_complains, marginalized_complains])

                kl = self.kl_div(a=self.target_dist,
                                 b=gd)

                kl_rollout.append(self.decay_factor**step * kl)

                if step < (self.n_step_per_rollout - 1):

                    y = np.random.random() < marginalized_complains.item()
                    b = self.revise_belief(y=y, b=self.b, x=self.x)

        return np.sum(kl_rollout)

    def free_energy_beliefs(self, b_prime, y, x, b):
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
            b=self.generative_density(y=y, x=x, b=b))

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
