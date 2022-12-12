import torch
import torchviz


def update_target_positions(shift, n_target, window_size):
    window_width = window_size[0]
    x_shift = shift * window_width
    area_width = window_width/n_target
    pos = torch.zeros((n_target, 3))
    for i in range(n_target):
        x = area_width * i + x_shift
        x = torch.where(x >= window_width, x - window_width, x)
        exceed = x + area_width - window_width
        first_width = area_width - torch.max(torch.zeros(1), exceed)
        second_width = torch.where(exceed > 0, area_width - first_width, torch.zeros(1))
        pos[i, 0] = x
        pos[i, 1] = first_width
        pos[i, 2] = second_width
    return pos


def update_fish_position(fish_position, fish_jump, window_size):
    new_pos = fish_position + fish_jump
    for coord in range(2):
        new_pos[coord] = torch.where(new_pos[coord] < 0, 0, new_pos[coord])
        new_pos[coord] = torch.where(new_pos[coord] > window_size[coord], window_size[coord], new_pos[coord])
        # new_pos[coord] = torch.clip(new_pos[coord].clone(), min=0, max=self.env.size(coord))
    return new_pos


def act(fish_position, target_positions, goal, sigma, jump_size, window_size):

    noise = torch.randn(2) * sigma
    mu = compute_mu(
        fish_position=fish_position,
        target_positions=target_positions,
        goal=goal,
        jump_size=jump_size)

    fish_jump = mu + noise

    # new_pos = update_fish_position(fish_position=fish_position, fish_jump=fish_jump, window_size=window_size)
    # fish_jump = new_pos - fish_position
    # return fish_jump
    return fish_jump


def aim(target_positions, goal, fish_position, jump_size):

    def compare_with(position):
        first_diff = (x - x_fish) ** 2
        second_diff = (position - x_fish)**2
        return torch.where(first_diff < second_diff, x, position)

    x_fish = fish_position[0]
    x = target_positions[goal, 0]
    first_width = target_positions[goal, 1]
    second_width = target_positions[goal, 2]

    x_center = torch.where(second_width > 0, compare_with(second_width), compare_with(x + first_width))

    x_mvt = torch.where(x_center > x_fish, jump_size, - jump_size)

    return torch.tensor([x_mvt, 0.], requires_grad=True)


def compute_mu(target_positions, goal, fish_position, jump_size):

    x = target_positions[goal, 0]
    first_width = target_positions[goal, 1]
    second_width = target_positions[goal, 2]

    fish_x = fish_position[0]
    fish_is_in = x <= fish_x <= x+first_width or (second_width > 0. and 0. <= fish_x <= second_width)
    # print("fish_is_in", fish_is_in)
    # mu = torch.where(fish_is_in,
    #                  torch.zeros(2, requires_grad=True),
    #                  aim(target_positions=target_positions, goal=goal, fish_position=fish_position,
    #                      jump_size=jump_size))
    # where(fish_is_in, 40, -40)  # torch.tensor([x, first_width], requires_grad=True)
    # print(torch.where(x > 0.5, 40, -40))
    return torch.where(x > 0.5, torch.tensor([40]), torch.tensor([-40]))

def logp_action(target_positions, fish_jump, fish_initial_position, window_size, sigma,
                jump_size):

    n_target = target_positions.shape[0]
    logp = []
    for goal in range(n_target):
        mu = compute_mu(
            target_positions=target_positions,
            goal=goal,
            fish_position=fish_initial_position, jump_size=jump_size)
        logp_goal = 0
        for coord in range(2):
            a_coord = fish_jump[coord]
            border = window_size[coord]
            dist = torch.distributions.Normal(mu[coord], sigma)
            logp_coord = torch.where(0 < a_coord < border, dist.log_prob(a_coord), torch.zeros(1))
            logp_coord = torch.where(a_coord <= 0, torch.log(1 - dist.cdf(a_coord)), logp_coord)
            logp_coord = torch.where(a_coord >= border,  torch.log(dist.cdf(a_coord)), logp_coord)
            logp_goal += logp_coord
        logp.append(logp_goal)
    logp = torch.concat(logp)
    return logp


def main():

    jump_size = 0.02

    window_size = torch.ones(2)
    sigma = 0.2

    b = torch.ones(2)
    fish_position = torch.tensor([0.3, 0.5])

    a_n_epoch = 2
    b_n_epoch = 2
    a_lr = 0.1
    b_lr = 0.1

    unc_a = torch.nn.Parameter(torch.tensor([0.5]))
    a_opt = torch.optim.Adam([unc_a, ], lr=a_lr)

    for epoch in range(a_n_epoch):

        # old_unc_a = unc_a.clone()
        a_opt.zero_grad()

        action = torch.sigmoid(unc_a)
        print("ACTION", action)

        # Sample the user goal --------------------------
        q = torch.softmax(b - b.max(), dim=0)
        goal = torch.distributions.Categorical(probs=q).sample()

        # -----------------------------------------------

        fish_position_rol = fish_position.clone()
        b_rol = b.clone()

        # ---- Update positions based on action ---------------------------------------------
        targets_positions_rol = update_target_positions(shift=action, n_target=b.shape[0],
                                                        window_size=window_size)
        print("targets_positions", targets_positions_rol)

        # ------------------------------------------------------------------------------
        # Evaluate epistemic value -----------------------------------------------------
        # ------------------------------------------------------------------------------

        # Simulate action based on goal ----------------------------------------

        fish_jump = act(target_positions=targets_positions_rol, goal=goal,
                        fish_position=fish_position_rol,
                        sigma=sigma,
                        jump_size=jump_size,
                        window_size=window_size)

        print("fish_jump", fish_jump)
        #
        # # Compute log probability of user action given a specific goal in mind -------
        # logp_y = logp_action(
        #     target_positions=targets_positions_rol,
        #     fish_initial_position=fish_position_rol,
        #     fish_jump=fish_jump,
        #     sigma=sigma,
        #     window_size=window_size,
        #     jump_size=jump_size)
        #
        # print("logp_y", logp_y)
        #
        # # logq = torch.log_softmax(b_rol, dim=0)
        # logq = torch.log_softmax(b_rol - b_rol.detach().max(), dim=0)
        # logp_yq = logq + logp_y
        #
        # fish_position = update_fish_position(
        #     fish_position=fish_position, fish_jump=fish_jump, window_size=window_size)
        #
        # # Revise belief -------------------
        #
        # b_rol = torch.nn.Parameter(b_rol)
        # b_opt = torch.optim.Adam([b_rol, ], lr=b_lr)
        #
        # q_rol = None
        # kl_div = None
        # for _ in range(b_n_epoch):
        #
        #     old_b = b_rol.clone()
        #     b_opt.zero_grad()
        #     q_rol = torch.softmax(b_rol - b_rol.detach().max(), dim=0)
        #     kl_div = torch.sum(q_rol * (q_rol.log() - logp_yq))
        #     kl_div.backward(retain_graph=True)
        #     b_opt.step()
        #
        #     if torch.isclose(old_b, b_rol).all():
        #         break
        #
        # print("KL div",  kl_div.requires_grad)
        #
        # epistemic_value = kl_div
        # print("epistemic value", epistemic_value.detach().numpy())
        #
        # # --------------------------------------
        # # Compute extrinsic value
        # # --------------------------------------
        #
        # entropy = - (q_rol * q_rol.log()).sum()
        # extrinsic_value = - entropy
        # print("extrinsic value", extrinsic_value.detach().numpy())

        # --------------------------------------
        # Compute loss
        # --------------------------------------

        action_loss = fish_jump.sum()   # - extrinsic_value  # - epistemic_value - extrinsic_value
        torchviz.make_dot(action_loss).render("debug", format="png")
        action_loss.backward(retain_graph=True)
        print("unconstrained action GRAD", unc_a.grad)
        a_opt.step()
        print("unconstrained action", unc_a.data)

        # if torch.isclose(old_unc_a, unc_a).all():
        #     break

    return torch.sigmoid(unc_a).detach()


if __name__ == "__main__":
    main()
