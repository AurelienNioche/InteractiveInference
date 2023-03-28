import itertools
import numpy as np

import jax
import jax.numpy as jnp

jax_key = jax.random.PRNGKey(42)
print(f"JAX devices: {jax.devices()}")
print(f"JAX device type: {jax.devices()[0].device_kind}")

def jax_safelog(x):
  return jnp.log( jnp.maximum(x, 1e-16 ))

def jax_step_fun(o, q, πs, p_o, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value and state information gain."""
    # update belief from new observation
    joint = q * p_o[:,o]
    q = joint / joint.sum()
    # policy rollout
    step = lambda q, a: (q @ p_t[:,a,:], q @ p_t[:,a,:])
    qs_π = lambda q, π: jax.lax.scan(step, init=q, xs=π)[1]
    qs_πs = jax.vmap(qs_π, in_axes=(None, 0), out_axes=(0))
    q_ss = qs_πs(q, πs)
    # pragmatic value
    pragmatic = (q_ss @ log_p_c).sum(axis=1)
    # state info gain
    p_oo = q_ss @ p_o # prior
    joint = q_ss[...,None] * p_o
    q_oo = joint / joint.sum( axis=2, keepdims=True ) # conditional
    d_oo = (q_oo * (jax_safelog( q_oo ) - jax_safelog( q_ss )[...,None])).sum( axis=2 ) # KL
    info_gain = (d_oo * p_oo).sum(axis=(1, 2)) # sum over o and t
    # action selection
    π = πs[jnp.argmax(pragmatic + info_gain)]
    # propagate belief through time
    q = q @ p_t[:,π[0],:]
    return q, π

jit_step_fun = jax.jit(jax_step_fun)

class MinimalAgentJax:
    """ Minimal agent performing exact inference in fully discrete POMDPs"""
    def __init__(self, 
                 env,
                 target_state, 
                 k=2): # sample plan (False), select max negEFE (True).
        self.env = env
        self.target_state = target_state
        self.k = k
        self.p_t = jnp.asarray(env.p_s1_given_s_a)
        self.p_o = jnp.asarray(env.p_o_given_s)
        self.πs = jnp.asarray( 
          [x for x in itertools.product( range(self.env.a_N), repeat=self.k )]
        )
        
    def reset(self):
        # initialize state preference
        q_star = np.eye(self.env.s_N)[self.target_state] \
                 + 1/(self.env.s_N*5) * np.ones(self.env.s_N)
        self.q_star = q_star / q_star.sum()
        self.log_p_c = jnp.log( self.q_star )
        # initialize state prior as uniform
        self.q = jnp.asarray(np.ones(self.env.s_N) / self.env.s_N )
    
    def step(self, o, use_jit=True):
        params = {
            'o': o,
            'q': self.q,
            'p_o': self.p_o,
            'p_t': self.p_t,
            'log_p_c': self.log_p_c,
            'πs': self.πs,
        }
        self.q, π = jit_step_fun(**params) if use_jit else jax_step_fun(**params)
        return π[0]