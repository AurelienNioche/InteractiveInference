import torch
from torchviz import make_dot


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
        return torch.softmax(self.b, dim=0)

    @property
    def np_belief(self):
        return self.belief.detach().numpy().copy()

    @property
    def action(self):
        return self.a

    def reset(self):
        self.b = torch.ones(self.n_target)
        self.a = torch.zeros(1)

    def revise_belief(self, b, fish_action, target_positions, fish_position):
        """
        Update the belief based on a new observation
        """
        logp_y = self.user_model.logp_action(
            target_positions=target_positions,
            fish_initial_position=fish_position,
            fish_jump=fish_action)

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

        b = self.b.detach().clone()

        fish_position = self.env.fish_position

        a_n_epoch = self.action_selection_max_epochs
        b_n_epoch = self.belief_update_max_epochs
        a_lr = self.action_selection_learning_rate
        b_lr = self.belief_update_learning_rate

        sim_act = self.user_model.act
        logp_action = self.user_model.logp_action
        update_target_positions = self.env.update_target_positions
        update_fish_position = self.env.update_fish_position

        unc_a = torch.nn.Parameter(torch.tensor([0.4537]))
        a_opt = torch.optim.Adam([unc_a, ], lr=0.1)

        for _ in range(a_n_epoch):

            old_unc_a = unc_a.clone()
            a_opt.zero_grad()

            # # -----------------------------------
            # # Build action plans
            # # -----------------------------------
            #
            action = torch.sigmoid(unc_a)
            print("ACTION", action)
            # action_plan = torch.zeros((n_rollout, n_step_per_rollout))
            # action_plan[:, 0] = first_action
            # if n_step_per_rollout > 1:
            #     action_plan[:, 1:] = torch.rand((n_rollout, n_step_per_rollout - 1))

            # action_loss = 0
            # for rol in range(n_rollout):

            # efe_rollout = 0

            # Sample the user goal --------------------------

            q = torch.softmax(b - b.max(), dim=0)
            print("q", q)
            goal = torch.distributions.Categorical(probs=q).sample()

            print("goal", goal)

            # -----------------------------------------------

            fish_position_rol = fish_position.clone()
            b_rol = b.clone()

            # for step in range(n_step_per_rollout):

            # action = action_plan[rol, step]

            # ---- Update positions based on action ---------------------------------------------
            targets_positions_rol = update_target_positions(shift=action)
            print("target_positions", targets_positions_rol.requires_grad)

            # ------------------------------------------------------------------------------
            # Evaluate epistemic value -----------------------------------------------------
            # ------------------------------------------------------------------------------

            # Simulate action based on goal ----------------------------------------

            fish_jump = sim_act(target_positions=targets_positions_rol, goal=goal,
                                fish_position=fish_position_rol)
            print("fish jump", fish_jump.requires_grad)

            # Compute log probability of user action given a specific goal in mind -------
            logp_y = logp_action(target_positions=targets_positions_rol, fish_jump=fish_jump,
                                 fish_initial_position=fish_position_rol)

            print("logp_y", logp_y.requires_grad)

            logq = torch.log_softmax(b_rol, dim=0)
            # logq = torch.log_softmax(b_rol - b_rol.detach().max(), dim=0)
            logp_yq = logq + logp_y

            print("logp_yq", logp_yq.requires_grad)

            fish_position = update_fish_position(fish_position=fish_position, fish_jump=fish_jump)

            # Revise belief -------------------

            b_rol = torch.nn.Parameter(b_rol)
            b_opt = torch.optim.Adam([b_rol, ], lr=b_lr)

            q_rol = None
            kl_div = None
            for _ in range(b_n_epoch):

                old_b = b_rol.clone()
                b_opt.zero_grad()
                q_rol = torch.softmax(b_rol - b_rol.detach().max(), dim=0)
                kl_div = torch.sum(q_rol * (q_rol.log() - logp_yq))
                kl_div.backward(retain_graph=True)
                b_opt.step()

                if torch.isclose(old_b, b_rol).all():
                    break

            print("kl div",  kl_div.requires_grad)

            epistemic_value = kl_div
            print("epistemic value", epistemic_value.detach().numpy())

            # --------------------------------------
            # Compute extrinsic value
            # --------------------------------------

            extrinsic_value = (q_rol * q_rol.log()).sum()  # minus entropy

            # --------------------------------------
            # Compute loss
            # ---------------------------------------
            #efe_step = - epistemic_value - extrinsic_value
            # efe_step = - first_action
            # efe_rollout += decay_factor ** step * efe_step

            # action_loss += efe_rollout
            # action_loss /= n_rollout

            action_loss = - epistemic_value - extrinsic_value
            make_dot(action_loss).render("debug", format="png")
            action_loss.backward()
            print(unc_a.grad)
            a_opt.step()
            print("unc_a", unc_a.data)

            if torch.isclose(old_unc_a, unc_a).all():

                break

        return torch.sigmoid(unc_a).detach()
