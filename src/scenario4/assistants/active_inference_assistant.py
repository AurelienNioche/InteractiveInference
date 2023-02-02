import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.special import softmax

import jupyter_core
import jupyterlab


class Assistant:

    """
    x: target positions
    psi: latent state; preferences of the user for each target
    actions: moving one target closer, and moving away all the other targets
    b_star: preferences of the assistant; preferences relate to the distance to the preferred target
    """

    def __init__(self,
                 fish_model,
                 belief_update_learning_rate=0.1,
                 belief_update_max_epochs=500,
                 action_selection_max_epochs=500,
                 decision_rule='active_inference',
                 decision_rule_parameters=None):

        super().__init__()

        self.fish_model = fish_model
        self.n_target = fish_model.n_target

        self.belief_update_max_epochs = belief_update_max_epochs
        self.belief_update_learning_rate = belief_update_learning_rate

        self.action_selection_max_epochs = action_selection_max_epochs

        self.decision_rule = decision_rule
        if decision_rule_parameters is None:
            if decision_rule in ("random", "static", "active_inference"):
                decision_rule_parameters = dict()
            else:
                raise ValueError("Decision rule not recognized")
        self.decision_rule_parameters = decision_rule_parameters

        self.a = None  # Current action
        self.b = torch.ones(self.n_target)  # Beliefs over user preferences

    @property
    def belief(self):
        return torch.softmax(self.b, dim=0)

    @property
    def np_belief(self):
        return self.belief.detach().numpy().copy()

    @property
    def action(self):
        return self.a

    def revise_belief(self, b, fish_jump, target_positions, fish_initial_position, screen_size):
        """
        Update the belief based on a new observation
        """
        logp_y = torch.from_numpy(self.fish_model.logp_action(
            target_positions=target_positions,
            fish_initial_position=fish_initial_position,
            fish_jump=fish_jump,
            screen_size=screen_size))

        logq = torch.log_softmax(b - b.max(), dim=0)
        logp_yq = (logq + logp_y)

        # Start by creating a new belief `b_prime` from the previous belief `b`
        b_prime = torch.nn.Parameter(b.clone())

        loss = None

        opt = torch.optim.Adam([b_prime, ], lr=self.belief_update_learning_rate)

        # Minimise free energy
        for epoch in range(self.belief_update_max_epochs):

            old_b_prime = b_prime.clone()
            opt.zero_grad()
            q_prime = torch.softmax(b_prime - b_prime.detach().max(), dim=0)
            loss = (q_prime * (q_prime.log() - logp_yq)).sum()
            loss.backward()
            opt.step()

            if torch.isclose(old_b_prime, b_prime).all():
                break

        return b_prime.detach(), loss.item()

    def act(self,
            fish_jump,
            previous_target_positions,
            previous_fish_position,
            fish_position,
            update_target_positions,
            screen_size):

        self.b, _ = self.revise_belief(fish_jump=fish_jump, b=self.b.detach(),
                                       target_positions=previous_target_positions,
                                       fish_initial_position=previous_fish_position,
                                       screen_size=screen_size)

        print("Belief over preferences", torch.softmax(self.b, dim=0))

        self.a = getattr(self, f'_act_{self.decision_rule}')(
            fish_position=fish_position,
            update_target_positions=update_target_positions,
            screen_size=screen_size,
            **self.decision_rule_parameters)
        return self.a

    def _act_random(self, *args, **kwargs):
        return np.random.random()

    def _act_static(self, *args, **kwargs):
        return 0

    def _act_active_inference(self, fish_position, update_target_positions, screen_size):

        actions = sorted(np.random.random(10))
        losses = [self.loss_action(
            action=a,
            fish_position=fish_position,
            update_target_positions=update_target_positions,
            screen_size=screen_size) for a in actions]
        p_action = softmax(-np.asarray(losses))
        rd_act = np.random.choice(actions, p=p_action, size=1000)
        plt.hist(rd_act)
        plt.show()
        action = actions[np.argmin(losses)]

        print("DECISION TAKEN", "*" * 100)
        print("ACTION", action)
        print("*" * 100)
        return action

    # @staticmethod
    # def sigmoid(x):
    #     return 1/(1 + np.exp(-x))

    def loss_action(self, action, fish_position, update_target_positions, screen_size,
                    n_goal_sample=100):

        # action = self.sigmoid(actions[0])
        # fish_position, = args

        b = self.b

        # Sample the user goal ---------------------------------------------------------

        q = torch.softmax(b - b.max(), dim=0)
        goals = torch.distributions.Categorical(probs=q).sample((n_goal_sample, ))
        loss_sum = 0
        for goal in goals:

            fish_position_rol = fish_position.copy()
            b_rol = b.clone()

            # ---- Update positions based on action ----------------------------------------

            targets_positions_rol = update_target_positions(shift=action, screen_size=screen_size)

            # ------------------------------------------------------------------------------
            # Evaluate epistemic value -----------------------------------------------------
            # ------------------------------------------------------------------------------

            # Simulate action based on goal ------------------------------------------------

            fish_jump = self.fish_model.act(
                target_positions=targets_positions_rol,
                goal=goal,
                fish_position=fish_position_rol,
                screen_size=screen_size)

            b_rol, kl_div = self.revise_belief(
                b=b_rol,
                fish_initial_position=fish_position_rol,
                fish_jump=fish_jump,
                target_positions=targets_positions_rol,
                screen_size=screen_size)

            epistemic_value = kl_div

            # --------------------------------------
            # Compute extrinsic value
            # --------------------------------------

            # q_rol = torch.softmax(b_rol - b_rol.max(), dim=0)
            # entropy = - (q_rol * q.log()).sum()
            # extrinsic_value = entropy.item()

            extrinsic_value = np.log(int(self.fish_model.fish_is_in(
                target_positions=targets_positions_rol,
                fish_position=fish_position_rol, goal=goal)) + 1e-16)

            # fish_x = fish_position[0]
            # x, first_width, second_width, height = targets_positions_rol[goal]
            # screen_width, screen_height = screen_size
            # target_width = screen_width/self.n_target
            # if second_width > first_width:
            #     center = second_width - target_width/2
            # else:
            #     center = x + target_width/2
            #
            # dist_to_center = (center - fish_x)**2
            #
            # extrinsic_value = - dist_to_center

            # --------------------------------------
            # Compute loss
            # --------------------------------------

            loss = - extrinsic_value - epistemic_value
            loss_sum += loss

        loss_avg = loss_sum / n_goal_sample
        print("action", action, "loss", loss_avg)
        return loss_avg
