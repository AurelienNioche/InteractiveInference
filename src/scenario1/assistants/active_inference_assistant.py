import numpy as np
import torch


def distance(pos1, pos2):
    return np.sqrt(((pos1 - pos2) ** 2).sum())


class Assistant:

    """
    x: target positions
    psi: latent state; preferences of the user for each target
    actions: moving one target closer, and moving away all the other targets
    b_star: preferences of the assistant; preferences relate to the distance to the preferred target
    """

    def __init__(self,
                 n_target,
                 user_model,
                 window,
                 constant_amplitude=3,
                 seed=123,
                 belief_update_learning_rate=0.1,
                 belief_update_max_epochs=500,
                 action_selection_learning_rate=0.1,
                 action_selection_max_epochs=500,
                 decision_rule='active_inference',
                 decision_rule_parameters=None):

        super().__init__()

        self.window = window
        self.constant_amplitude = constant_amplitude

        self.n_target = n_target
        self.user_model = user_model

        self.belief_update_max_epochs = belief_update_max_epochs
        self.belief_update_learning_rate = belief_update_learning_rate

        self.action_selection_learning_rate = action_selection_learning_rate
        self.action_selection_max_epochs = action_selection_max_epochs

        self.decision_rule = decision_rule
        if decision_rule_parameters is None:
            if decision_rule == "active_inference":
                decision_rule_parameters = dict(
                    decay_factor=0.9,
                    n_rollout=5,
                    n_step_per_rollout=2,
                    n_sample_epistemic_value=10)
            elif decision_rule == "random":
                decision_rule_parameters = dict()
            else:
                raise ValueError("Decision rule not recognized")
        self.decision_rule_parameters = decision_rule_parameters

        self.a = None  # Current action
        self.b = None  # Beliefs over user preferences
        self.x = None  # State of the world (positions of the targets)

        self.user_action = None

    @property
    def belief(self):
        return torch.softmax(self.b, dim=0).detach().numpy().copy()

    @property
    def action(self):
        return self.a.to_numpy()

    def reset(self):

        if self.n_target == 1:
            self.x = torch.tensor([[0.5, 0.5]])

        elif self.n_target == 2:
            self.x = torch.tensor([[0.25, 0.25], [0.75, 0.75]])
        else:
            for i in range(self.n_target):
                self.x[i] = torch.rand(2)

        for i in range(self.n_target):
            self.x[i] *= self.window.size()

        self.a = torch.rand(self.n_target) * 360
        self.b = torch.ones(self.n_target)
        return self.x

    def revise_belief(self, y, b, x):
        """
        Update the belief based on a new observation
        """

        logp_y = torch.from_numpy(self.user_model.p_action(positions=x, action=y))

        logq = torch.log_softmax(b - b.max(), dim=0)
        logp_yq = logq + logp_y
        logp_yq.requires_grad = True

        # Start by creating a new belief `b_prime` from the previous belief `b`
        b_prime = torch.nn.Parameter(b.clone())

        loss = None

        opt = torch.optim.Adam([b_prime, ], lr=self.belief_update_learning_rate)

        # Minimise free energy
        for step in range(self.belief_update_max_epochs):

            old_b_prime = b_prime.clone()

            opt.zero_grad()

            q_prime = torch.softmax(b_prime - b_prime.max(), dim=0)
            loss = torch.sum(q_prime * (q_prime.log() - logp_yq))

            loss.backward()
            opt.step()

            if torch.isclose(old_b_prime, b_prime).all():
                # print(f"converged at step {step}")
                break

        return b_prime.detach(), loss.item()

    def act(self, user_action):

        self.user_action = user_action,
        self.b, _ = self.revise_belief(y=user_action, b=self.b, x=self.x)

        self.a = getattr(self, f'_act_{self.decision_rule}')(
            **self.decision_rule_parameters)
        self.x = self.update_environment(x=self.x, a=self.a)
        return self.x.to_numpy()

    def _act_random(self):

        angles = torch.rand(self.n_target) * 360
        return angles

    def _act_active_inference(self,
                              decay_factor,
                              n_rollout,
                              n_step_per_rollout,
                              n_sample_epistemic_value):

        # Start by creating a new belief `b_prime` from the previous belief `b`
        a = torch.nn.Parameter(self.a.clone())

        opt = torch.optim.Adam([a, ], lr=self.belief_update_learning_rate)

        # Minimise free energy
        for step in range(self.belief_update_max_epochs):

            old_a = a.clone()
            opt.zero_grad()

            action = torch.sigmoid(a)*360
            loss = self.rollout(
                action=action,
                n_rollout=n_rollout,
                n_step_per_rollout=n_step_per_rollout,
                decay_factor=decay_factor,
                n_sample_epistemic_value=n_sample_epistemic_value)  # kl div

            loss.backward()
            opt.step()

            if torch.isclose(old_a, a).all():
                break

        return a.detach()

    def rollout(self, action, n_rollout, n_step_per_rollout, decay_factor,
                n_sample_epistemic_value):

        total_efe = 0
        for _ in range(n_rollout):
            efe_rollout = 0

            x = self.x.clone()
            b = self.b.clone()

            action_plan = np.zeros(n_step_per_rollout, dtype=int)
            action_plan[0] = action
            if n_step_per_rollout > 1:
                action_plan[1:] = torch.rand(size=(n_step_per_rollout-1, self.n_target)) * 360

            for step, action in enumerate(action_plan):

                prev_x = x.copy()
                x = self.update_environment(x=x, a=action)

                efe_step, b = self.efe(x=x, prev_x=prev_x,
                                       b=b, n_sample_epistemic_value=n_sample_epistemic_value)

                efe_rollout += decay_factor**step * efe_step

            total_efe += efe_rollout.item()

        total_efe /= n_rollout
        return total_efe

    def efe(self, prev_x, x, b, n_sample_epistemic_value):

        # --- Compute epistemic value
        epistemic_value, b_new = self.epistemic_value(
            b=b, x=x, prev_x=prev_x,
            n_sample_epistemic_value=n_sample_epistemic_value)

        # --- Compute extrinsic value
        # objective_dist = torch.distributions.HalfNormal(scale_objective)
        # log_p = (objective_dist.log_prob(x) * b).sum()
        extrinsic_value = 0  # log_p

        efe_step = - extrinsic_value - epistemic_value

        return efe_step, b_new

    def epistemic_value(self, b, x, prev_x, n_sample_epistemic_value):

        ep_value = torch.zeros(n_sample_epistemic_value)
        b_new = torch.zeros((n_sample_epistemic_value, len(self.b)))

        goals = torch.multinomial(self.b, n_sample_epistemic_value)
        user_actions = torch.tensor([self.user_model.sim_act(
            goal=goal,
            prev_positions=prev_x,
            positions=x) for goal in goals])

        for i, y in enumerate(user_actions):
            b_new_i, kl_div_i = self.revise_belief(y=y, b=b, x=x)
            b_new[i] = b_new_i
            ep_value[i] = kl_div_i

        epistemic_value = ep_value.mean()
        b_new = b_new[torch.randint(low=0, high=n_sample_epistemic_value, size=1)]
        return epistemic_value, b_new

    def update_environment(self, x, a):
        pos = x
        for i in range(self.n_target):
            angle = a[i]
            # 0° Move left
            # 90° Move down
            # 180° Move right
            # 270° Move up
            x_prime = 1.0
            if 90 < angle <= 270:
                x_prime *= -1

            y_prime = torch.tan(torch.deg2rad(angle)) * x_prime

            norm = self.constant_amplitude / torch.sqrt(y_prime**2 + x_prime**2)
            movement = torch.tensor([x_prime, y_prime]) * norm
            pos[i] += movement

        max_coord = torch.from_numpy(self.window.size())
        for coord in range(2):
            pos[:, coord] = torch.clip(pos[:, coord], 0, max_coord[coord])

        return pos

    @property
    def target_positions(self):
        return self.x.to_numpy()
