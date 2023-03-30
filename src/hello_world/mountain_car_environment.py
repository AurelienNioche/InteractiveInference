from collections import namedtuple # forward_ode()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # show_p_s()
from scipy.integrate import solve_ivp # forward_ode()
from tqdm import tqdm # transition_dynamics()


class MountainCarDiscrete():
    
    n_x = 32
    n_v = 32 # number of bins along each state dimension
    n_s = n_x * n_v
    bounds_x = np.array([-2,2])
    bounds_v = np.array([-3,3])
    xx = np.linspace(-2, 2, n_x) 
    vv = np.linspace(-3, 3, n_v)
    aa = np.array([-2, -1, 0, 1, 2]) # actions
    dynamics_filename = 'mountain_car_transition_dynamics.npy'
    
    @staticmethod
    def force_a(a):
        """ force due to action 
        a (np.ndarray): continuous valued action. Scalar or vector of scalars.
        """
        return np.tanh(a) 
    
    @staticmethod
    def force_g(x):
        """ force due to gravity 
        x (np.ndarray): continuous valued horizontal position. Scalar or vector of scalars.
        """
        phi = np.zeros_like(x)
        is_leq0 = x<=0
        phi_1 = 2*x+1
        a = (1 + 5*x**2)
        phi_2 = a**(-0.5)
        phi_3 = - 5*x**2*a**(-1.5)
        phi_4 = (x/2)**4 # negative of paper
        phi = phi_2 + phi_3 + phi_4
        phi[is_leq0] = phi_1[is_leq0]  
        return -phi
    
    @staticmethod
    def force_friction(v):
        """ friction force due to inertia 
        v (np.ndarray): continuous valued velocity. Scalar or vector of scalars.
        """
        return -1/4*v
    
    @classmethod
    def heights(cls):
        gg = cls.force_g(cls.xx)
        hh = np.cumsum(-gg)/cls.n_x
        return hh - hh.min()
    
    @classmethod
    def render_env(cls):
        """ Render properties of the environment."""
        xx, vv = cls.xx, cls.vv
        
        fig, ax = plt.subplots(1, 4, figsize=(4*8, 6))
        plt.sca(ax[0])
        plt.plot(xx, cls.heights(), label='height')
        plt.xlabel('position')
        plt.ylabel('height')
        plt.legend()

        plt.sca(ax[1])
        plt.plot(xx, cls.force_g(xx), label='gravity')
        plt.xlabel('position')
        plt.ylabel('gravitational force')
        plt.legend()

        plt.sca(ax[2])
        aa = np.linspace(-2, 2, 101)
        plt.plot(xx, cls.force_a(xx), label='action')
        plt.xlabel('a')
        plt.ylabel('action force')
        plt.legend()

        plt.sca(ax[3])
        plt.plot(vv, cls.force_friction(vv), label='inertia')
        plt.xlabel('velocity')
        plt.ylabel('friction force')
        plt.legend()

        plt.show()
    
    @staticmethod
    def index(x, bounds, nbins):
        r = (bounds[1]-bounds[0]) + 1e-6
        idx = ((x-bounds[0])/r*nbins).astype(int)
        return idx

    @staticmethod
    def value(idx, bounds, nbins):
        r = (bounds[1]-bounds[0])
        w = r / nbins
        return float(idx)/nbins * r + bounds[0] +w/2

    @classmethod
    def index_x(cls, x):
        return cls.index(x, cls.bounds_x, cls.n_x)

    @classmethod
    def value_x(cls, x):
        return cls.value(x, cls.bounds_x, cls.n_x)

    @classmethod
    def index_v(cls, v):
        return cls.index(v, cls.bounds_v, cls.n_v)

    @classmethod
    def value_v(cls, v):
        return cls.value(v, cls.bounds_v, cls.n_v)

    @classmethod
    def idx_s_from_idx_xv(cls, x, v):
        return v * cls.n_x + x

    @classmethod
    def idx_xv_from_idx_s(cls, s):
        x = s % cls.n_x
        v = (s-x) / cls.n_x
        return [int(x), int(v)]

    @classmethod
    def s_from_index_s(cls, i_s):
        i_x, i_v = cls.idx_xv_from_idx_s(i_s)
        return [cls.value_x(i_x), cls.value_v(i_v)]

    @classmethod
    def index_s_from_s(cls, s):
        return cls.idx_s_from_idx_xv(cls.index_x(s[0]), cls.index_v(s[1]))
    
    @classmethod
    def ode(cls, t, s, a):
        dx = s[1]
        f_a = cls.force_a(a)
        f_g = cls.force_g(s[0])
        f_i = cls.force_friction(s[1])
        dv = f_a + f_g + f_i 
        #print(f't {t:.2f}, s {s[0,0]:.2f}, {s[1,0]:.2f}, a {a:.2f}, fa {f_a:.2f}, fg {f_g[0]:.2f}, fi {f_i[0]:.2f}')
        return np.array([dx, dv])
    
    @classmethod
    def forward_ode(cls, s, a, debug=False, use_ivp=True):
        if use_ivp:
            sol = solve_ivp(cls.ode, 
                            t_span=[0.0, 0.5], 
                            y0=s, 
                            rtol=1e-6, 
                            vectorized=True, 
                            args=(a,)) # Fig 5: [0, 0.5] with zero action start
        else:
            # Euler update
            dt = 0.6
            ds = f(None, np.array(s)[:,np.newaxis], a)
            sol = namedtuple('sol', 'y')
            sol.y = (np.array(s)+ds.squeeze()*dt).reshape((-1,1))

        return sol if debug else sol.y[:,-1]
    
    
    @classmethod
    def load_transition_dynamics(cls, filename=None):
        filename = filename if filename is not None else cls.dynamics_filename
        print(f'Loading transition dynamics from file "{filename}"')
        with open(filename, 'rb') as f:
            τ = np.load(f)
            return τ
        
    @classmethod
    def save_transition_dynamics(cls, τ, filename=None):
        filename = filename if filename is not None else cls.dynamics_filename
        with open(filename, 'wb') as f:
            np.save(f, τ)
    
    @classmethod
    def transition_dynamics(cls, n_samples=10_000):
        print(f'Computing transition dynamics from simulation of {n_samples} samples.')
        # construct table
        n_x, n_v, bounds_x, bounds_v, aa = cls.n_x, cls.n_v, cls.bounds_x, cls.bounds_v, cls.aa
        n_s = n_x * n_v
        cell_x = (bounds_x[1]-bounds_x[0])/n_x
        cell_v = (bounds_v[1]-bounds_v[0])/n_v
        n_a = aa.shape[0]
        p_forward = np.zeros(shape=(aa.shape[0], n_s, n_s))
        # sample states s0
        bounds = np.array([bounds_x, bounds_v])
        ss0 = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size=(n_samples,2))

        for a_i, a in enumerate(cls.aa):
            # compute successor states
            ss1 = []
            for s0 in tqdm(ss0):
                ss1.append(cls.forward_ode(s=s0, a=a))

            ss1 = np.array(ss1)
            # handle position out of bounds with velocity in same direction
            out_of_bounds_min = np.array(ss1[:,0] <= bounds_x[0]) * np.array(ss1[:,1] < 0)
            out_of_bounds_max = np.array(ss1[:,0] >= bounds_x[1]) * np.array(ss1[:,1] > 0)
            out_of_bounds = np.maximum(out_of_bounds_min, out_of_bounds_max)
            ss1[:,1] = (1-out_of_bounds) * ss1[:,1] # set velocity to zero
            ss1[:,0] = np.clip(ss1[:,0], bounds_x[0], bounds_x[1])
            # handle velocity out of bounds
            ss1[:,1] = np.clip(ss1[:,1], bounds_v[0], bounds_v[1])
            # compute state indices
            ss0_i = cls.index_s_from_s(ss0.T)
            ss1_i = cls.index_s_from_s(ss1.T)
            # build transition matrix
            for s0_i, s1_i in zip(ss0_i, ss1_i):
                p_forward[a_i, s0_i, s1_i] += 1  
        
        #normalise
        for a_i in range(cls.aa.shape[0]):
            p_forward[a_i] /= p_forward[a_i].sum(axis=1, keepdims=True)
        
        p_forward = p_forward.swapaxes(0,1)
        return p_forward
    
    def __init__(self, load_transition_dynamics=True):
        self.a_N = len(self.aa)
        self.s_N = self.n_s
        
        if load_transition_dynamics:
            τ = self.load_transition_dynamics()
        else:
            τ = self.transition_dynamics()
            
        self.p_s1_given_s_a = τ # Matrix B
        self.hh = self.heights() # used in plot_trajectory
        
    def reset(self):
        self.s_i = self.index_s_from_s(np.array([0, 0]))
        return self.s_i
        
    def step(self, a):
        self.s_i = np.random.choice(self.s_N, p=self.p_s1_given_s_a[self.s_i, a])
        return self.s_i
        
    def _height(self, state):
        x, N = self.xx, self.n_x
        pos = np.clip(state[0], x[0], x[-1])
        r = x[-1] - x[0] + 1e-6
        i_x = int((pos-x[0])/r * N)
        #print(pos, r, i_x, h[i_x])
        return self.hh[i_x]
        
    def plot_trajectory(self, states, **kwargs):
        states = np.array([self.s_from_index_s(s) for s in states])
        return self.plot_trajectory_continuous(states, **kwargs)
        
    def plot_trajectory_continuous(self, states, ax=None, color=None, alpha=1, label=None, legend=True):
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(2*10, 5))
        
        xx = [s[0] for s in states]
        vv = [s[1] for s in states]
        hh = np.array([self._height(s) for s in states])
        #hh += np.linspace(0, 1, hh.shape[0]) * 0.00

        plt.sca(ax[0])
        plt.plot(xx, vv, 'x-', label=label, color=color, alpha=alpha)
        plt.plot([xx[0]], [vv[0]], 'g*', ms=10, alpha=alpha, label='start')
        plt.plot([xx[-1]], [vv[-1]], 'r*', ms=10, alpha=alpha, label='end')
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.title('state transition')
        plt.grid('on')
        if legend:
            plt.legend()

        plt.sca(ax[1])
        plt.plot(self.xx, self.heights(), label='surface')
        plt.plot(xx, hh, 'r-')
        plt.scatter([xx[-1]], [hh[-1]], color='red', marker='o', label='car')
        if legend:
            plt.legend()
            
    def show_p_s(self, p_s, ax=None, cmap='gray'):
        if ax is not None:
            plt.sca(ax)
        else:
            ax = plt.gca()
            
        n_v, bounds_x, bounds_v = self.n_v, self.bounds_x, self.bounds_v

        im = plt.imshow(np.flip(p_s.reshape(n_v, -1), axis=0), 
                        cmap=cmap, 
                        extent=bounds_x.tolist() + bounds_v.tolist(), 
                        aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)