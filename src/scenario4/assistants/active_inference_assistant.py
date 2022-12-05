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
                 user_model,
                 belief_update_learning_rate=0.1,
                 belief_update_max_epochs=500,
                 action_selection_learning_rate=0.1,
                 action_selection_max_epochs=500,
                 decision_rule='active_inference',
                 decision_rule_parameters=None):

        super().__init__()

        self.env = user_model.env
        self.n_target = self.env.n_target
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
                    n_rollout=1,
                    n_step_per_rollout=1)
            elif decision_rule == "random":
                decision_rule_parameters = dict()
            else:
                raise ValueError("Decision rule not recognized")
        self.decision_rule_parameters = decision_rule_parameters

        self.a = None  # Current action
        self.b = None  # Beliefs over user preferences

    @property
    def belief(self):
        return torch.softmax(self.b, dim=0).detach().numpy().copy()

    @property
    def action(self):
        return self.a.numpy().copy()

    def reset(self):
        self.b = torch.ones(self.n_target)

    def revise_belief(self, b, fish_action, target_positions, fish_position):
        """
        Update the belief based on a new observation
        """
        logp_y = torch.from_numpy(
            self.user_model.logp_action(target_positions=target_positions, own_position=fish_position,
                                        action=fish_action))

        logq = torch.log_softmax(b - b.max(), dim=0).detach()
        logp_yq = (logq + logp_y).detach()

        # Start by creating a new belief `b_prime` from the previous belief `b`
        b_prime = torch.nn.Parameter(b.clone())

        loss = None

        opt = torch.optim.Adam([b_prime, ], lr=self.belief_update_learning_rate)

        # Minimise free energy
        for step in range(self.belief_update_max_epochs):

            old_b_prime = b_prime.clone()
            opt.zero_grad()
            q_prime = torch.softmax(b_prime - b_prime.detach().max(), dim=0)
            loss = (q_prime * (q_prime.log() - logp_yq)).sum()
            loss.backward()
            opt.step()

            if torch.isclose(old_b_prime, b_prime).all():
                break

        return b_prime.detach(), loss.detach()

    def act(self, fish_action, previous_target_positions, previous_fish_position):

        self.b, _ = self.revise_belief(fish_action=fish_action, b=self.b.detach(),
                                       target_positions=previous_target_positions,
                                       fish_position=previous_fish_position)

        self.a = getattr(self, f'_act_{self.decision_rule}')(
            **self.decision_rule_parameters)
        return self.a

    def _act_random(self):
        return torch.rand(1)

    def _act_active_inference(self,
                              decay_factor,
                              n_rollout,
                              n_step_per_rollout):

        x = self.x.detach().clone()
        b = self.b.detach().clone()
        a = self.a.detach().clone()

        a_n_epoch = self.action_selection_max_epochs
        b_n_epoch = self.belief_update_max_epochs
        a_lr = self.action_selection_learning_rate
        b_lr = self.belief_update_learning_rate

        user_sigma = self.user_model.sigma
        n_target = self.n_target
        mvt_amplitude = self.constant_amplitude
        max_coord = self.window.size()

        def update_environment(positions, action):

            for i in range(n_target):
                angle = action[i]
                if 90 < angle <= 270:
                    x_prime = -1
                else:
                    x_prime = 1
                y_prime = torch.tan(torch.deg2rad(angle)) * x_prime

                norm = mvt_amplitude / torch.sqrt(y_prime ** 2 + x_prime ** 2)
                movement = torch.tensor([x_prime, y_prime]) * norm
                positions[i] += movement

            for coord in range(2):
                for target in range(n_target):
                    if positions[target, coord] > max_coord[coord]:
                        positions[target, coord] = max_coord[coord]

            return positions

        def logp_action(positions, action, prev_positions):

            logp_y = torch.zeros(n_target)
            for target in range(n_target):
                for coord in range(2):
                    d = positions[target, coord] - prev_positions[target, coord]
                    logp_coord = torch.distributions.Normal(d, user_sigma).log_prob(action[coord])
                    logp_y[target] += logp_coord

            return logp_y

        sim_act = self.user_model.sim_act

        a = torch.nn.Parameter(a)
        a_opt = torch.optim.Adam([a, ], lr=a_lr)

        for _ in range(a_n_epoch):

            old_a = a.clone()
            a_opt.zero_grad()

            # -----------------------------------
            # Build action plans
            # -----------------------------------

            first_action = torch.sigmoid(a)

            action_plan = torch.zeros((n_rollout, n_step_per_rollout, n_target))
            action_plan[:, 0] = first_action
            if n_step_per_rollout > 1:
                action_plan[:, 1:, :] = torch.rand((n_rollout, n_step_per_rollout - 1, n_target)) * 360

            action_plan *= 360  # Convert in degrees

            total_efe = 0
            for rol in range(n_rollout):

                efe_rollout = 0

                # Sample the user goal --------------------------

                q = torch.softmax(b - b.max(), dim=0)
                goal = torch.multinomial(q, 1)[0]

                # -----------------------------------------------

                x_rol = x.clone()
                b_rol = b.clone()

                for step in range(n_step_per_rollout):

                    action = action_plan[rol, step]

                    # ---- Update positions based on action ---------------------------------------------

                    x_rol_prev = x_rol.clone()

                    x_rol = update_environment(positions=x_rol, action=action)

                    # ------------------------------------------------------------------------------
                    # Evaluate epistemic value -----------------------------------------------------
                    # ------------------------------------------------------------------------------

                    # Simulate action based on goal ----------------------------------------

                    y = sim_act(positions=x_rol, goal=goal, prev_positions=x_rol_prev)

                    # Compute log probability of user action given a specific goal in mind -------
                    logp_y = logp_action(positions=x_rol, action=y, prev_positions=x_rol_prev)

                    logq = torch.log_softmax(b_rol - b_rol.detach().max(), dim=0)
                    logp_yq = logq + logp_y

                    # Revise belief -------------------

                    b_rol = torch.nn.Parameter(b_rol)
                    b_opt = torch.optim.Adam([b_rol, ], lr=b_lr)

                    q_rol, kl_div = None, None

                    for _ in range(b_n_epoch):

                        old_b = b_rol.clone()
                        b_opt.zero_grad()
                        q_rol = torch.softmax(b_rol - b_rol.detach().max(), dim=0)
                        kl_div = torch.sum(q_rol * (q_rol.log() - logp_yq))
                        kl_div.backward(retain_graph=True)
                        b_opt.step()

                        if torch.isclose(old_b, b_rol).all():
                            break

                    epistemic_value = kl_div

                    # --------------------------------------
                    # Compute extrinsic value
                    # --------------------------------------

                    extrinsic_value = (q_rol * q_rol.log()).sum()  # minus entropy

                    # --------------------------------------
                    # Compute loss
                    # ---------------------------------------
                    efe_step = - epistemic_value - extrinsic_value
                    efe_rollout += decay_factor ** step * efe_step

                total_efe += efe_rollout

            total_efe /= n_rollout
            total_efe.backward()
            a_opt.step()

            if torch.isclose(old_a, a).all():
                break

        return a.detach()

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
            for target in range(self.n_target):
                if pos[target, coord] > max_coord[coord]:
                    pos[target, coord] = max_coord[coord]

        return pos

    @property
    def target_positions(self):
        return self.x.numpy()
