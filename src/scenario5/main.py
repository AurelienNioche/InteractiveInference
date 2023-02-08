import scipy
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt


p_backward = np.load('p_backward.npy')

n_x, n_v = 32, 32
n_s = n_x * n_v
bounds_x = np.array([-2,2])
bounds_v = np.array([-3,3])
cell_x = (bounds_x[1]-bounds_x[0])/n_x
cell_v = (bounds_v[1]-bounds_v[0])/n_v

def index(x, bounds, nbins):
    r = (bounds[1]-bounds[0]) + 1e-6
    idx = ((x-bounds[0])/r*nbins).astype(int)
    return idx

def value(idx, bounds, nbins):
    r = (bounds[1]-bounds[0])
    w = r / nbins
    return float(idx)/nbins * r + bounds[0] +w/2

def index_x(x):
    return index(x, bounds_x, n_x)

def value_x(x):
    return value(x, bounds_x, n_x)

def index_v(v):
    return index(v, bounds_v, n_v)

def value_v(v):
    return value(v, bounds_v, n_v)

def idx_s_from_idx_xv(x, v):
    return v * n_x + x

def idx_xv_from_idx_s(s):
    x = s % n_x
    v = (s-x) / n_x
    return [int(x), int(v)]

def s_from_index_s(i_s):
    i_x, i_v = idx_xv_from_idx_s(i_s)
    return [value_x(i_x), value_v(i_v)]

def index_s_from_s(s):
    return idx_s_from_idx_xv(index_x(s[0]), index_v(s[1]))


def s1_given_s_a(s, a, debug=False, use_ivp=True):
    if use_ivp:
        sol = scipy.integrate.solve_ivp(f, t_span=[0.0, 0.5], y0=s, rtol=1e-6, vectorized=True,
                                        args=(a,))  # Fig 5: [0, 0.5] with zero action start
    else:
        dt = 0.6
        ds = f(None, np.array(s)[:, np.newaxis], a)
        sol = namedtuple('sol', 'y')
        sol.y = (np.array(s) + ds.squeeze() * dt).reshape((-1, 1))

    if debug:
        return sol

    return sol.y[:, -1]


def f_friction(v):
    return -1 / 4 * v

def f_action(a):
    return np.tanh(a)


def f(t, s, a):
    dx = s[1]
    f_a = f_action(a)
    f_g = f_gravity(s[0])
    f_i = f_friction(s[1])
    dv = f_a + f_g + f_i
    # print(f't {t:.2f}, s {s[0,0]:.2f}, {s[1,0]:.2f}, a {a:.2f}, fa {f_a:.2f}, fg {f_g[0]:.2f}, fi {f_i[0]:.2f}')
    return np.array([dx, dv])


def f_gravity(x):
    phi = np.zeros_like(x)
    is_leq0 = x <= 0
    phi_1 = 2 * x + 1
    a = (1 + 5 * x ** 2)
    phi_2 = a ** (-0.5)
    phi_3 = - 5 * x ** 2 * a ** (-1.5)
    phi_4 = (x / 2) ** 4  # negative of paper
    phi = phi_2 + phi_3 + phi_4
    phi[is_leq0] = phi_1[is_leq0]
    return -phi


N = 101
x = np.linspace(-2, 2, N)
g = f_gravity(x)
w = 4 / N
h = np.cumsum(-g) * w
h -= h.min()

s0 = [-0.06451613, -0.09677419]
aa_fig4 = [4, 4, 3, 2, 2, 1, 1, 2, 3, 3, 3, 1, 2, 1, 2]
# , 4, 4, 3, 2, 2, 2, 2, 2, 3, 3, 2, 1, 2, 1, 2]
aa_fig5 = [2, 1, 1, 3, 3, 3, 2, 2, 3, 1, 2, 2, 2, 2, 2, 1]  # [2, 1, 1, 3, 3, 3, 2, 2, 3, 1, 2, 2, 2, 2, 2, 1]

aa = aa_fig5
a = [-2, -1, 0, 1, 2]
aa = [a[x] for x in aa]

ss = [s0]
ss_full = [s0]
for a in aa:
    sol = s1_given_s_a(s=ss[-1], a=a, debug=True)
    ss.append(sol.y[:, -1])
    ss_full.extend(sol.y.T.tolist())


def normalize(p_transition, mode='column'):
    # sum over elements in each row, shaped to normalize correctly
    row_norm = lambda x: np.maximum(x.sum(axis=1).reshape((-1, 1)), 1e-6)
    # sum over elements in each column, shaped to normalize correctly
    column_norm = lambda x: np.maximum(x.sum(axis=0), 1e-6)

    if mode == 'column':
        # print('normalize for each column to sum to 1.')
        norm = column_norm
    else:
        # print('normalize for each row to sum to 1.')
        norm = row_norm

    if len(p_transition.shape) > 2:
        # print('interpreting axis 0 as action, normalizing each 2D subtensor')
        for a_i in range(p_transition.shape[0]):
            # normalize column-wise
            p = p_transition[a_i]
            l1 = norm(p)
            p = p / l1
            p_transition[a_i] = p

        return p_transition

    else:
        l1 = norm(p_transition)
        return p_transition / l1


def height(state):
    pos = np.clip(state[0], x[0], x[-1])
    r = x[-1] - x[0] + 1e-6
    i_x = int((pos - x[0]) / r * N)
    # print(pos, r, i_x, h[i_x])
    return h[i_x]


def plot_trajectory(states, ax=None, color=None, alpha=1, label=None, legend=True):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(2 * 10, 5))

    xx = [s[0] for s in states]
    vv = [s[1] for s in states]
    hh = np.array([height(s) for s in states])
    hh += np.linspace(0, 1, hh.shape[0]) * 0.00

    plt.sca(ax[0])
    plt.plot(xx, vv, 'x-', label=label, color=color, alpha=alpha)
    plt.plot([xx[0]], [vv[0]], 'g*', ms=10, label='start')
    plt.plot([xx[-1]], [vv[-1]], 'r*', ms=10, label='end')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.title('state transition')
    plt.grid('on')
    if legend:
        plt.legend()

    plt.sca(ax[1])
    plt.plot(x, h, label='surface')
    plt.plot(xx, hh, 'r-')
    plt.scatter([xx[-1]], [hh[-1]], color='red', marker='o', label='car')
    if legend:
        plt.legend()


def show_inference_result(bb_a, bb_s):
    t = 1
    n_samples = 20  # sample tranjectories from beliefs
    n_rows, n_cols = 4, 2
    p_forward = normalize(p_backward, mode='rows')

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))

    # explore actions
    # show belief over sequence of actions at convergence
    cax = ax[0][0]
    plt.sca(cax)
    plt.imshow(bb_a[-1].T, cmap='gray', aspect='auto')
    cax.set_title('belief over actions across future timesteps');

    # show convergence of belief over actions at one timestep
    cax = ax[0][1]
    plt.sca(cax)
    for a_i in range(n_a):
        plt.plot(np.array(bb_a)[:, t, a_i], label=f'a_{t}={[-2, -1, 0, 1, 2][a_i]}')
    cax.set_title('optimization of belief over next action');
    plt.xlabel('optimisation timestep')
    plt.ylabel('p(a_next)')
    plt.legend()

    # explore states
    # show belief over sequence of states at convergence
    cax = ax[1][0]
    plt.sca(cax)
    plt.imshow(np.array(bb_s)[-1, :, :].T, aspect='auto', interpolation='nearest')
    cax.set_title('belief over states across future timesteps');

    # show convergence of belief over states at one timestep
    cax = ax[1][1]
    plt.sca(cax)
    plt.imshow(np.array(bb_s)[:, t, :].T, aspect='auto', interpolation=None)
    cax.set_title('optimisation of belief over next state');

    # illustrate sequence of most likely states in phase space
    # sample trajectories from belief over future states
    cax = ax[2][0]
    for i in range(n_samples):
        ss = np.array([s_from_index_s(np.random.choice(range(n_s), p=np.array(bb_s)[-1, t, :])) for t in range(T)])
        plot_trajectory(ss, ax=[ax[2][0], ax[2][1]], legend=i == 0, color='gray', alpha=0.5)
    cax.set_title('sample trajectories from belief over future states');

    # sample trajectories from belief over future actions
    cax = ax[3][0]
    for i in range(n_samples):
        s_i = index_s_from_s([0, 0])
        ss = [s_from_index_s(s_i)]

        for t in range(1, T):
            a_i = np.random.choice(range(n_a), p=bb_a[-1][t])
            # _i = np.random.choice(range(n_a))
            p_s_i = p_forward[a_i, s_i, :]
            s_i = np.random.choice(n_s, p=p_s_i)
            ss.append(s_from_index_s(s_i))

        plot_trajectory(ss, ax=[ax[3][0], ax[3][1]], legend=i == 0, color='gray', alpha=0.5)

    cax.set_title('sample trajectories from belief over future actions')
    plt.show()


T = 16  # planning horizon
n_a = p_backward.shape[0]
aa = np.array([-2, -1, 0, 1, 2], dtype=int)


def slog(x):
    # save log
    sx = np.maximum(x, 1e-6)
    return np.log(sx / sx.sum())


# 1. compute all priors
ln_pa = slog(1 / n_a)
prior_s0 = np.eye(n_s)[index_s_from_s([0, 0])]
ln_p0 = slog(prior_s0)
prior_sT = np.eye(n_s)[index_s_from_s([1, 0])]
ln_pT = slog(prior_sT)
ln_p = slog(p_backward)

msg_s_right = np.zeros((T, n_s))
msg_s_right[-1] = ln_pT
backtrack_a = np.zeros((T, n_s), dtype=int)
backtrack_s = np.zeros((T, n_s), dtype=int)

# forward pass
for t in reversed(range(T - 1)):
    arg = ln_p + msg_s_right[t + 1] + ln_pa
    # take max over action only
    max_a = np.max(arg, axis=0)
    # identify next state associated with the maximum for each s0
    argmax_s1 = np.argmax(max_a, axis=1)  # backtracking s1
    # take max over next state only
    max_s1 = np.max(arg, axis=2)
    # identify action associated with the maximum for each s0
    argmax_a = np.argmax(max_s1, axis=0)  # backtracking a
    # take max over both state and action
    max_a_s1 = np.max(max_a, axis=1)  # outgoing message
    # debug only: identify max s0
    argmax_s0 = np.argmax(max_a_s1)

    msg_s_right[t] = max_a_s1
    backtrack_a[t] = argmax_a
    backtrack_s[t] = argmax_s1

    print('t', t,
          's0', s_from_index_s(argmax_s0),
          'a', aa[argmax_a[argmax_s0]],
          's1', s_from_index_s(argmax_s1[argmax_s0]),
          'message', max_a_s1[argmax_s0],
          'argument', arg[argmax_a[argmax_s0], argmax_s0, argmax_s1[argmax_s0]],
          'validation', arg.max())

print("Backtracking")

# backtracking
s0 = [0, 0]  # initial state
s0_i = index_s_from_s(s0)  # index of initial state
eye_s = np.eye(n_s)
eye_a = np.eye(n_a)

ss_i = [s0_i]
print(s_from_index_s(ss_i[-1]))
aa_i = [0]
for t in range(T - 1):
    a_next = backtrack_a[t, ss_i[-1]]
    s_next = backtrack_s[t, ss_i[-1]]
    ss_i.append(s_next)
    aa_i.append(a_next)
    print('action', a_next, 'next state', s_from_index_s(s_next))

bb_s = np.array([[eye_s[s_i] for s_i in ss_i]])
bb_a = np.array([[eye_a[a_i] for a_i in aa_i]])

show_inference_result(bb_a, bb_s)