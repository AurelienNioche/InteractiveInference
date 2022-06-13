import numpy as np

from . assistants import Assistant


class AiAssistant(Assistant):

    def __init__(self, n_targets, beta, learning_rate,
                 n_epochs=500,
                 *args, **kwargs):

        super().__init__(n_targets, *args, **kwargs)

        self.beta = beta

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        self.b_star = ...
        self.actions = ...

        self.b = None
        self.mu_hat = None
        self.t = None

    def step(self, action: np.ndarray):

        s = action  # sensory state is action of other user

        # Pick action state, a \in {-1, 0, 1}
        # Calculate the free energy given my target (intent) distribution, current state distribution, & sensory input
        # Do this for all (three) actions and select the action with minimum free energy.
        i = np.argmin([self.free_energy(self.b_star, self.b, s, a) for a in self.actions])
        a = self.actions[i]

        # Update my internal state.
        # Start by creating a belief for my new position b_prime, from my belief for my previous position b
        # by mixing according to movement probability given action a:
        b_prime = self.b.copy()
        # Note, this is not essential, as the minimisation below will update b_prime,
        # but it does provide a good starting point for gradient descent

        # Now minimise free energy
        for step in range(self.n_epochs):
            b_prime -= self.learning_rate * self.fe_gradient(b_prime, self.b, s, a)

        # Save position, sensory output, and internal state for plotting
        self.b = b_prime

        observation = a
        done = False

        reward, info = None, None
        return observation, reward, done, info

    def reset(self):
        super().reset()
        self.mu_hat = np.zeros(self.n_targets)
        self.t = 1  # Need to be one more than the user

        self.b = np.zeros(self.n_targets)

    def model_encoding_derivative(self, b):
        """
        Derivative of the model encoding for free energy gradient calculation
        """
        softmax_b = self.model_encoding(b)
        # Softmax derivative
        return np.diag(softmax_b) - np.outer(softmax_b, softmax_b)

    def fe_gradient(self, b_star, b, s, a):
        """
        Partial derivatives of the free energy with respect to belief R1.
        """
        J = self.model_encoding_derivative(b_star)
        Y = np.log(self.variational_density(b_star) / self.generative_density(b, a, s))
        return np.dot(J, Y)

    def sensory_dynamics(self, s):
        """
        P(s | psi)
        Probability of experiencing sensory state s given current state psi.
        """

        # update user model
        alpha = 1 / self.t

        self.mu_hat[:] = self.mu_hat * (1 - alpha) + self.pos * alpha

        pred_user_action = self.beta * (self.mu_hat - self.pos)

        sd = np.zeros(self.n_targets)
        sd[:] = pred_user_action - s == 0.

        assert np.sum(sd) == 1.

        self.t += 1

        return sd

    @staticmethod
    def model_encoding(b):
        """
        Probability of each target to be the preferred target
        as encoded in the internal state.
        """
        # Softmax function. The shift by b.max() is for numerical stability
        return np.exp(b - b.max()) / np.sum(np.exp(b - b.max()))

    def generative_density(self, b, a, s):
        """
        P(psi', s | b, a)
        Agent's prediction of its new position given its internal state and selected action
        (calculated separately for two sensory states).

        P(psi', s | b, a) = Sum_over_psi(P(psi' | s, b, a, psi) * P(s | b, a, psi) * P(psi | b, a))
                          = Sum_over_psi(P(psi' | a, psi) * P(s | psi) * P(psi | b))

        since psi' only depends on a and psi, s only depends on psi, and psi only depends on b.

        """
        # sensory dynamics for each position:
        """P(s | psi)"""
        sd = self.sensory_dynamics(s)  # Array: number of possible states (psi)

        """
        P(psi', s | b, a) = Sum_over_psi(P(psi' | a, psi) * P(s | psi) * P(psi | b))
        Note that the Sum_over_psi is only taken over the two positions psi that can result in getting to psi' given a
        """

        return None  # Array giving the probability of psi prime and s given a
        # (size is number of possible beliefs)

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
        return np.sum(a * (np.log(a) - np.log(b)))

    def free_energy(self, b_star, b, s, a):
        """
        KL divergence between variational density and generative density for a fixed
        sensory state s.
        """
        return self.KL(self.variational_density(b_star),
                       self.generative_density(b=b, a=a, s=s))
