import numpy as np
import torch
import gym


class AiAssistant(gym.Env):

    """
    x: position of the targets
    psi: latent state; preferences of the user for each target
    actions: moving one target closer, and moving away all the other targets
    b_star: preferences of the assistant; preferences relate to the distance to the preferred target
    """

    def __init__(self, n_targets, beta, learning_rate,
                 starting_x=0.5,
                 step_size=0.1,
                 min_x=0.0,
                 max_x=1.0,
                 n_epochs=500,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.n_targets = n_targets

        self.step_size = step_size

        self.beta = beta

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        self.actions = torch.arange(n_targets)  # With target to move

        self.dist_space = torch.linspace(0, 1, int(1/step_size)+1)

        self.min_x = min_x
        self.max_x = max_x

        self.starting_x = starting_x

        self.b = None
        self.t = None
        self.x = None

    def reset(self):

        self.b = torch.ones(self.n_targets)
        self.x = torch.ones(self.n_targets) * self.starting_x
        observation = self.x  # np.random.choice(self.actions)
        return observation  # reward, done, info can't be included

    @property
    def b_star(self):
        """
        b_star is about the distance to the desired target
        The closer, the better
        :return:
        """
        return torch.tensor([1, 0])
    # def b_star(self, x, scale=0.05):
    #     """
    #     b_star is about the distance to the desired target
    #     The closer, the better
    #     :return:
    #     """
    #     return torch.distributions.HalfNormal(scale).log_prob(x+0.00001)

    def step(self, action: np.ndarray):

        print("Initial belief", self.variational_density(self.b))

        y = action  # sensory state is action of other user

        # Update my internal state.
        # Start by creating a new belief b_prime, from my belief for the previous belief b
        b_prime = torch.nn.Parameter(self.b.clone())

        self.b.requires_grad = True

        opt = torch.optim.Adam([b_prime, ], lr=self.learning_rate)
        # Now minimise free energy
        for step in range(self.n_epochs):
            opt.zero_grad()
            loss = self.free_energy_beliefs(b_prime=b_prime, b=self.b, y=y, x=self.x)
            loss.backward()
            opt.step()

        # Update internal state
        self.b = b_prime.detach()

        print("Belief after revision", self.variational_density(self.b))

        # Pick an action
        # Calculate the free energy given my target (intent) distribution, current state distribution, & sensory input
        # Do this for all actions and select the action with minimum free energy.
        kl_div = [self.free_energy_action(
            b_star=self.b_star, b=self.b, x=self.x, a=a) for a in self.actions]

        print("kl_div", kl_div)

        i = np.argmin(kl_div)
        a = self.actions[i]

        self.x = self.take_action(x=self.x, a=a)

        print("action", a)

        observation = self.x.numpy()
        done = False

        reward, info = None, None
        return observation, reward, done, info

    def take_action(self, x, a):

        step_size_other = self.step_size/(self.n_targets-1)

        mask = np.ones(self.n_targets)
        mask[a] = 0

        x[mask] += step_size_other
        x[a] -= self.step_size
        x[x > self.max_x] = self.max_x
        x[x < self.min_x] = self.min_x

        return x

    def generative_density(self, b, y, x):
        """
        Q(s | o)
        """

        p_preferred = self.variational_density(b)

        like = torch.tanh(x * self.beta)

        p = like * p_preferred
        p /= torch.sum(p)

        if y:
            return p
        else:
            return 1-p

    @staticmethod
    def variational_density(b):
        """
        P(psi | b)
        Agent's belief about the external states (i.e. its current position in the
        world) or intention (i.e. desired position in the world) as encoded in the
        internal state.
        """
        # Softmax function. The shift by b.max() is for numerical stability
        b_scaled = b - b.max()
        return torch.nn.functional.softmax(b_scaled, dim=0)

    # @staticmethod
    # def KL(a, b):
    #     """
    #     Kullback-Leibler divergence between densities a and b.
    #     """
    #     # If access to distributions "registered" for KL div, then:
    #     # torch.distributions.kl.kl_divergence(p, q)
    #     # Otherwise:
    #     # torch.nn.functional.kl_div(a, b)
    #     return torch.sum(a * (torch.log(a) - torch.log(b)))

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

    def free_energy_action(self, b_star, b, x, a):
        """
        KL divergence between variational density and generative density for a fixed
        sensory state s.
        """
        p_preferred = self.variational_density(b)
        # return 1 - p_preferred[0], p_preferred[a],
        x = self.take_action(x=x.clone(), a=a)

        #for p_i, x_i in zip(p_preferred, x):
        joint = torch.tanh(x * self.beta) * p_preferred
        marginalized = joint.sum()

        gd = torch.tensor([1 - marginalized, marginalized])
        kl = torch.nn.functional.kl_div(target=b_star,
                                        input=gd)
        print("could be x", x)
        print("b_star", b_star)
        print("gd", gd)
        print("kl", kl)
        return kl

    def free_energy_beliefs(self, b_prime, b, y, x):
        """
        KL divergence between variational density and generative density for a fixed
        sensory state s.
        """
        return torch.nn.functional.kl_div(
            target=self.variational_density(b_prime),
            input=self.generative_density(b=b, y=y, x=x))
