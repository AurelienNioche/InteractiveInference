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

        self.b = None
        self.t = None
        self.x = None

    def reset(self):

        self.b = torch.ones(self.n_targets)
        self.x = torch.ones(self.n_targets)
        observation = torch.ones(self.n_targets)   # np.random.choice(self.actions)
        return observation  # reward, done, info can't be included

    @property
    def b_star(self):
        """
        b_star is about the distance to the desired target
        The closer, the better
        :return:
        """
        return torch.distributions.HalfNormal(0.1).log_prob(self.dist_space+0.01)

    def step(self, action: np.ndarray):

        s = action  # sensory state is action of other user

        # Pick an action
        # Calculate the free energy given my target (intent) distribution, current state distribution, & sensory input
        # Do this for all actions and select the action with minimum free energy.
        i = np.argmin([self.free_energy_action(self.b_star, self.b, s, a) for a in self.actions])
        a = self.actions[i]

        # Update my internal state.
        # Start by creating a new belief b_prime, from my belief for the previous belief b
        b_prime = torch.nn.Parameter(self.b.clone())

        self.b.requires_grad = True

        opt = torch.optim.Adam([b_prime, ], lr=self.learning_rate)
        # Now minimise free energy
        for step in range(self.n_epochs):
            opt.zero_grad()
            loss = self.free_energy_beliefs(b_prime, self.b, s, a)
            loss.backward()
            opt.step()

        # Update internal state
        self.b = b_prime.detach()

        self.x[:] += self.step_size
        self.x[a] -= 2*self.step_size
        self.x[self.x > self.max_x] = self.max_x
        self.x[self.x < self.min_x] = self.min_x

        print("belief", self.b)
        print("action", a)
        print("x", self.x)

        observation = self.x.numpy()
        done = False

        reward, info = None, None
        return observation, reward, done, info

    @staticmethod
    def model_encoding(b):
        """
        Probability of each target to be the preferred target
        as encoded in the internal state.
        """
        # Softmax function. The shift by b.max() is for numerical stability
        b_scaled = b - b.max()
        return torch.nn.functional.softmax(b_scaled, dim=0)

    def generative_density(self, b, a, s):
        """
        Q_b(s | a)
        """
        x_temp = self.x.clone()
        x_temp[:] -= self.step_size
        x_temp[a] += 2*self.step_size
        x_temp[x_temp > self.max_x] = self.max_x
        x_temp[x_temp < self.min_x] = self.min_x

        prior = 1 - self.variational_density(b)
        like = torch.tanh(x_temp[torch.arange(self.n_targets)] * self.beta)
        like /= torch.sum(like)
        p = like * prior
        p /= torch.sum(p)
        if s:
            return p
        else:
            return 1-p

    def variational_density(self, b):
        """
        P(psi | b)
        Agent's belief about the external states (i.e. its current position in the
        world) or intention (i.e. desired position in the world) as encoded in the
        internal state.
        """
        return self.model_encoding(b)

    @staticmethod
    def KL(a, b):
        """
        Kullback-Leibler divergence between densities a and b.
        """
        # If access to distributions "registered" for KL div, then:
        # torch.distributions.kl.kl_divergence(p, q)
        # Otherwise:
        # torch.nn.functional.kl_div(a, b)
        return torch.sum(a * (torch.log(a) - torch.log(b)))

    def free_energy_action(self, b_star, b, s, a):
        """
        KL divergence between variational density and generative density for a fixed
        sensory state s.
        """
        p = self.generative_density(b=b, a=a, s=s)

        x_temp = self.x.clone()
        x_temp[:] -= self.step_size
        x_temp[a] += 2*self.step_size
        x_temp[x_temp > self.max_x] = self.max_x
        x_temp[x_temp < self.min_x] = self.min_x

        dist = torch.zeros(len(self.dist_space))
        for i, x_ in enumerate(self.dist_space):
            for j in range(self.n_targets):
                p_j, x_j = p[j], x_temp[j]
                if np.isclose(x_j, x_):
                    dist[i] += p_j

        return self.KL(b_star,
                       dist)

    def free_energy_beliefs(self, b_prime, b, s, a):
        """
        KL divergence between variational density and generative density for a fixed
        sensory state s.
        """
        return self.KL(self.variational_density(b_prime),
                       self.generative_density(b=b, a=a, s=s))
