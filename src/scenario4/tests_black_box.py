import torch
import numpy as np
from scipy import optimize, stats


def update_target_positions(shift, n_target, window_size):
    window_width = window_size[0]
    x_shift = shift * window_width
    area_width = window_width/n_target
    pos = np.zeros((n_target, 3))
    for i in range(n_target):
        x = area_width * i + x_shift
        if x >= window_width:
            x -= window_width

        exceed = x + area_width - window_width
        first_width = area_width - max(0, exceed)
        if exceed > 0:
            second_width = area_width - first_width
        else:
            second_width = 0
        pos[i] = x, first_width, second_width
    return pos


def update_fish_position(fish_position, fish_jump, window_size):
    new_pos = fish_position + fish_jump
    for coord in range(2):
        new_pos[coord] = np.clip(new_pos[coord], a_min=0., a_max=window_size[coord])
    return new_pos


def act(fish_position, target_positions, goal, sigma, jump_size, window_size):

    noise = np.random.normal(size=2) * sigma
    mu = compute_mu(
        fish_position=fish_position,
        target_positions=target_positions,
        goal=goal,
        jump_size=jump_size)

    fish_jump = mu + noise

    new_pos = update_fish_position(fish_position=fish_position, fish_jump=fish_jump, window_size=window_size)
    fish_jump = new_pos - fish_position
    return fish_jump


def compute_aim(target_positions, goal, fish_position, jump_size):

    def compare_with(position):
        first_diff = (x - x_fish) ** 2
        second_diff = (position - x_fish)**2
        return np.where(first_diff < second_diff, x, position)

    x_fish = fish_position[0]
    x = target_positions[goal, 0]
    first_width = target_positions[goal, 1]
    second_width = target_positions[goal, 2]
    if second_width > 0:
        x_center = compare_with(second_width)
    else:
        x_center = compare_with(x + first_width)

    if x_center > x_fish:
        x_mvt = jump_size
    else:
        x_mvt = - jump_size

    aim = np.array([x_mvt, 0.])
    return aim


def compute_mu(target_positions, goal, fish_position, jump_size):

    x, first_width, second_width = target_positions[goal]

    fish_x = fish_position[0]
    fish_is_in = x <= fish_x <= x+first_width or (second_width > 0. and 0. <= fish_x <= second_width)
    if fish_is_in:
        mu = np.zeros(2)
    else:
        mu = compute_aim(
            target_positions=target_positions,
            goal=goal,
            fish_position=fish_position,
            jump_size=jump_size)

    return mu


def logp_action(
        target_positions, fish_jump, fish_initial_position, window_size, sigma, jump_size):

    n_target = target_positions.shape[0]
    logp = np.zeros(n_target)
    for goal in range(n_target):
        mu = compute_mu(
            target_positions=target_positions,
            goal=goal,
            fish_position=fish_initial_position, jump_size=jump_size)
        logp_goal = 0
        for coord in range(2):
            a_coord = fish_jump[coord]
            border = window_size[coord]
            dist = stats.norm(mu[coord], sigma)
            if 0 < a_coord < border:
                logp_goal += np.log(dist.pdf(a_coord))
            elif a_coord <= 0:
                logp_goal += np.log(1 - dist.cdf(a_coord))
            elif a_coord >= border:
                logp_goal += np.log(dist.cdf(a_coord))
            else:
                raise ValueError
        logp[goal] = logp_goal
    return logp


def loss_action(actions, *args):

    action = actions[0]
    b, fish_position, sigma, window_size, jump_size, b_lr, b_n_epoch = args

    # Sample the user goal --------------------------
    q = torch.softmax(b - b.max(), dim=0)
    goal = torch.distributions.Categorical(probs=q).sample()

    # -----------------------------------------------

    fish_position_rol = fish_position
    b_rol = b.clone()

    # ---- Update positions based on action ---------------------------------------------
    targets_positions_rol = update_target_positions(
        shift=action, n_target=b.shape[0], window_size=window_size)

    # ------------------------------------------------------------------------------
    # Evaluate epistemic value -----------------------------------------------------
    # ------------------------------------------------------------------------------

    # Simulate action based on goal ----------------------------------------

    fish_jump = act(target_positions=targets_positions_rol, goal=goal,
                    fish_position=fish_position_rol,
                    sigma=sigma,
                    jump_size=jump_size,
                    window_size=window_size)

    # Compute log probability of user action given a specific goal in mind -------
    logp_y = torch.from_numpy(logp_action(
        target_positions=targets_positions_rol,
        fish_initial_position=fish_position_rol,
        fish_jump=fish_jump,
        sigma=sigma,
        window_size=window_size,
        jump_size=jump_size))

    # logq = np.log_softmax(b_rol, dim=0)
    logq = torch.log_softmax(b_rol - b_rol.detach().max(), dim=0)
    logp_yq = logq + logp_y

    # fish_position = update_fish_position(
    #     fish_position=fish_position, fish_jump=fish_jump, window_size=window_size)

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
        kl_div.backward()
        b_opt.step()

        if torch.isclose(old_b, b_rol).all():
            break

    epistemic_value = kl_div

    # --------------------------------------
    # Compute extrinsic value
    # --------------------------------------

    entropy = - (q_rol * q_rol.log()).sum()
    extrinsic_value = - entropy

    # --------------------------------------
    # Compute loss
    # --------------------------------------

    loss = - epistemic_value - extrinsic_value
    return loss.item()


def main():

    jump_size = 0.02

    window_size = np.ones(2)
    sigma = 0.2

    b = torch.tensor([0.2, 1.3])
    fish_position = np.array([0.3, 0.5])

    b_n_epoch = 200
    b_lr = 0.1

    res = optimize.minimize(
        fun=loss_action, x0=np.zeros(1), bounds=[(0, 1)],
        args=(b, fish_position, sigma, window_size, jump_size, b_lr, b_n_epoch, ))

    action = res.x[0]
    print("action chosen", action)
    return action


if __name__ == "__main__":
    main()
