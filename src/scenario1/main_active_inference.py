import numpy as np
import sys
import pygame
import torch

from graphic.window import Window
from graphic.shape import Circle

from users.users import User

np.seterr(all='raise')


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
                 seed=123,
                 starting_x=0.5,
                 min_x=0.0,
                 max_x=1.0,
                 inference_learning_rate=0.1,
                 inference_max_epochs=500,
                 decision_rule='active_inference',
                 decision_rule_parameters=None):

        super().__init__()

        self.rng = np.random.default_rng(seed=seed)

        self.n_target = n_target
        self.user_model = user_model

        # Used for updating beliefs
        self.inference_max_epochs = inference_max_epochs
        self.inference_learning_rate = inference_learning_rate

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
        self.b = torch.ones(self.n_target)
        self.x = torch.ones(self.n_target) * self.starting_x
        observation = self.x  # np.random.choice(self.actions)
        return observation  # reward, done, info can't be included

    def revise_belief(self, y, b, x):
        """
        Update the belief based on a new observation
        """

        p = self.user_model.complain_prob(x)
        p_y = p ** y * (1 - p) ** (1 - y)

        q = torch.softmax(b - b.max(), dim=0)
        logp_yq = (p_y * q).log()
        logp_yq.requires_grad = True

        # Start by creating a new belief `b_prime` from the previous belief `b`
        b_prime = torch.nn.Parameter(b.clone())

        loss = None

        opt = torch.optim.Adam([b_prime, ], lr=self.inference_learning_rate)

        # Minimise free energy
        for step in range(self.inference_max_epochs):

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

    def act(self, user_action, environment_state):

        self.b, _ = self.revise_belief(y=user_action, b=self.b, x=environment_state)

        self.a = getattr(self, f'_act_{self.decision_rule}')(
            user_action=user_action,
            **self.decision_rule_parameters)
        return self.a

    def _act_random(self):

        angles = self.rng.uniform(0, 360, size=self.n_target)
        return angles

    def _act_active_inference(self,
                              decay_factor,
                              n_rollout,
                              n_step_per_rollout, efe='efe_on_obs'):

        efe = getattr(self, efe)

        kl_div = np.asarray([self.rollout(
            action=a,
            n_rollout=n_rollout,
            n_step_per_rollout=n_step_per_rollout,
            decay_factor=decay_factor,
            efe=efe,
        ) for a in self.actions])
        i = np.random.choice(np.nonzero(kl_div == np.min(kl_div))[0])

        # i = np.random.choice(np.arange(len(self.actions)),
        #                      p=torch.softmax(kl_div, dim=0).numpy())
        return i

    def rollout(self, action, n_rollout, n_step_per_rollout, decay_factor, efe):

        total_efe = 0
        for _ in range(n_rollout):
            efe_rollout = 0

            x = self.x.clone()
            b = self.b.clone()

            action_plan = np.zeros(n_step_per_rollout, dtype=int)
            action_plan[0] = action
            if n_step_per_rollout > 1:
                action_plan[1:] = self.rng.uniform(0, 360, size=(n_step_per_rollout-1, self.n_target))

            for step, action in enumerate(action_plan):

                self.update_target_positions(x=x, a=action)

                efe_step, p_y_under_q, b_new = efe(x=x, b=b)

                efe_rollout += decay_factor**step * efe_step

                if n_step_per_rollout > 1:
                    y = torch.bernoulli(p_y_under_q[1]).long()
                    b = b_new[y]

            total_efe += efe_rollout.item()

        total_efe /= n_rollout
        return total_efe

    def efe_on_obs(self, x, b):

        epistemic_value, p_y_under_q, b_new = self.epistemic_value(b, x)

        extrinsic_value = p_y_under_q[1] * torch.tensor([1e-8]).log()

        efe_step = - extrinsic_value - epistemic_value

        return efe_step, p_y_under_q, b_new

    def efe_on_latent(self, x, b,
                      scale_objective=0.05):

        # --- Compute epistemic value
        epistemic_value, p_y_under_q, b_new = self.epistemic_value(b, x)

        # --- Compute extrinsic value
        objective_dist = torch.distributions.HalfNormal(scale_objective)
        log_p = (objective_dist.log_prob(x) * b).sum()
        extrinsic_value = log_p

        efe_step = - extrinsic_value - epistemic_value

        return efe_step, p_y_under_q, b_new

    def epistemic_value(self, b, x):

        ep_value = torch.zeros(2)
        b_new = torch.zeros((2, len(self.b)))

        y = torch.arange(2)
        p = self.user_model.complain_prob(x)
        q = torch.softmax(b - b.max(), dim=0)
        p_y = p ** y.unsqueeze(dim=1) * (1 - p) ** (1 - y.unsqueeze(dim=1))
        p_y_under_q = (p_y * q).sum(1)

        for i in y:
            b_new_i, kl_div_i = self.revise_belief(y=i, b=b, x=x)
            b_new[i] = b_new_i
            ep_value[i] = kl_div_i

        epistemic_value = (p_y_under_q * ep_value).sum()
        return epistemic_value, p_y_under_q, b_new


class Display:

    def __init__(self, colors, hide_cursor=False,
                 control_per_artificial_user=False,
                 control_using_mouse=False):

        """
        :param colors: Iterable. Define the number of targets.
        :param hide_cursor: Bool.
        :param control_per_artificial_user: Bool.
        :param control_using_mouse: Bool.
        """

        self.fps = 30

        self.n_frame_selected = 20

        self.seed = 123

        self.mouse_scale = 0.5

        self.init_frames = 5

        # --- Graphic parameters ---

        self.hide_cursor = hide_cursor

        self.line_scale = 4.0
        self.base_radius = 20
        self.var_radius = 50
        self.width_circle_line = 2
        self.margin = 0.05

        # ---------------------------------------- #

        self.window = Window(fps=self.fps, hide_cursor=self.hide_cursor)

        # --------------------------------------- #

        self.n_target = len(colors)
        self.init_position = np.zeros((self.n_target, 2))
        self.colors = np.asarray(colors, dtype=object)
        self.color_selected = "red"

        # ---------------------------------------- #

        self.rng = np.random.default_rng(seed=self.seed)

        self.p_val = np.zeros(self.n_target)
        self.selected = np.zeros(self.n_target, dtype=bool)

        self.n_frame_since_selected = 0
        self.mouse_pos_prev_frame = np.zeros(2)
        self.control = np.zeros(2)
        self.pos = np.zeros((self.n_target, 2))

        # ------------------------------------------ #

        self.constant_amplitude = 3
        self.angle = np.zeros(self.n_target)

        self.movement = np.zeros((self.n_target, 2))

        # ------------------------------------------ #
        # Allow control (True/False)

        self.control_per_artificial_user = control_per_artificial_user
        self.control_using_mouse = control_using_mouse
        if self.control_using_mouse:
            assert self.control_using_mouse, "Mouse control should be on for the artificial user control to be on too"

        # ------------------------------------------ #

        self.initialize()
        self.reset()

    def initialize(self):

        for i in range(self.n_target):
            self.pos[i] = self.rng.random(2) * self.window.size()

        # To be sure that the mouse can be captured correctly...
        for i in range(self.init_frames):
            self.window.clear()
            self.window.update()

    def reset(self):

        self.p_val[:] = 0
        self.selected[:] = 0
        self.n_frame_since_selected = 0

        self.movement[:, 0] = self.constant_amplitude
        self.movement[:, 1] = 0

        self.window.move_back_cursor_to_the_middle()

    def implement_assistant_action(self, target_angles):

        for i in range(self.n_target):
            angle = target_angles[i]
            x, _ = self.movement[i]

            x_prime = np.abs(x)
            if 90 > angle < 270:
                x_prime *= -1

            y_prime = np.tan(np.radians(angle)) * x_prime

            norm = self.constant_amplitude / np.sqrt(y_prime**2 + x_prime**2)
            self.movement[i, :] = np.asarray([x_prime, y_prime]) * norm

        self.p_val[:] = 0
        self.selected[:] = 0
        self.pos[:] += self.movement

    def update_control(self, user_action):

        if self.control_per_artificial_user:
            self.window.move_mouse(movement=user_action)

        ctb = self.window.cursor_touch_border()

        if not ctb:
            mouse_pos = self.window.mouse_position
            delta_mouse = mouse_pos - self.mouse_pos_prev_frame
            self.mouse_pos_prev_frame[:] = mouse_pos
            add_to_ctl = delta_mouse * self.mouse_scale

        else:
            self.window.move_back_cursor_to_the_middle()
            self.mouse_pos_prev_frame[:] = self.window.mouse_position
            add_to_ctl = np.zeros(2)

        for coord in range(2):
            self.control[coord] += add_to_ctl[coord]

    def draw(self):

        visual_pos = np.zeros_like(self.pos)
        max_coord = self.window.size()
        for coord in range(2):
            vp = self.pos[:, coord] + self.control[coord]
            vp = np.clip(vp, 0, max_coord[coord])
            visual_pos[:, coord] = vp

        colors = np.copy(self.colors)
        colors[self.selected] = self.color_selected

        radius = self.base_radius + self.var_radius*self.p_val

        self.window.clear()

        for i in range(self.n_target):
            Circle(window=self.window,
                   position=visual_pos[i],
                   color=colors[i],
                   radius=radius[i]).draw()

        self.window.update()

    @staticmethod
    def check_keys():

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

    def update(self, user_action, assistant_action):

        self.check_keys()

        self.update_control(user_action)

        if self.selected.any():
            self.n_frame_since_selected += 1
            if self.n_frame_since_selected == self.n_frame_selected:
                self.reset()
        else:
            self.implement_assistant_action(assistant_action)

        self.draw()

        return self.pos

    @property
    def target_positions(self):
        return self.pos


def main():

    control_using_mouse = True
    control_per_artificial_user = False

    hide_cursor = False

    user_goal = 0
    user_sigma = 0.3
    user_alpha = 0.5

    colors = "orange", "blue"

    n_target = len(colors)

    display = Display(
        colors=colors,
        hide_cursor=hide_cursor,
        control_using_mouse=control_using_mouse,
        control_per_artificial_user=control_per_artificial_user)

    display.reset()
    assistant = Assistant(user_model=User, n_target=n_target,
                          decision_rule="active_inference")
    assistant.reset()
    user = User(n_target=n_target, sigma=user_sigma, goal=user_goal, alpha=user_alpha)
    user.reset()

    user_action = np.zeros(2)

    while True:

        assistant_action = assistant.act(
            user_action=user_action,
            environment_state=display.target_positions)
        display.update(user_action=user_action, assistant_action=assistant_action)
        if control_per_artificial_user:
            user_action = user.act(display.target_positions)
            user_action *= 2


if __name__ == "__main__":
    main()
