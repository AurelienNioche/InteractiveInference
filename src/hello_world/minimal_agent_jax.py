import subprocess as sp
import numpy as np

import jax
import jax.numpy as jnp

jax_key = jax.random.PRNGKey(42)
print(f"JAX devices: {jax.devices()}")
print(f"JAX device type: {jax.devices()[0].device_kind}")

# check GPU status
def get_gpu_memory():
    print(sp.check_output("nvidia-smi").decode('ascii'))
    
get_gpu_memory()

def jax_safelog(x):
  return jnp.log( jnp.maximum(x, 1e-16 ))

def jax_step_naive_fun(o, q, πs, p_o, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value and state information gain.
    
    This implementation is left here for illustrative purposes only. Optimised functions for production are below."""
    # update belief from new observation
    joint = q * p_o[:,o]
    q = joint / joint.sum()
    # policy rollout: 
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

jit_step_naive_fun = jax.jit(jax_step_naive_fun)

def jax_step_batched_fun(num_actions, o, q, πs, p_o, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value and state information gain."""
    # update belief from new observation
    joint = q * p_o[:,o]
    q = joint / joint.sum()
    
    # policy rollout: 
    # coalesced access to transition dynamics speeds up vectorised rollouts across policies
    def step(q, a):
        # iterate over values of a
        def step_a(i, q):
            is_a = (a==i)
            # weighted average between q and q' with binary weight
            q = (1-is_a) * q + is_a * (q @ p_t[:,i,:]) # replace with q + is_a (q @ p_t[:,i,:] - q)
            return q
        
        q = jax.lax.fori_loop(0, num_actions, step_a, q)
        return q, q
    
    qs_π = lambda q, π: jax.lax.scan(step, init=q, xs=π)[1]
    qs_πs = jax.vmap(qs_π, in_axes=(None, 0), out_axes=(0)) # carry, output
    
    # batching policies helps avoid GPU out of memory errors
    def batch_nefe(q, πs):
        q_ss = qs_πs(q, πs)
        pragmatic = (q_ss @ log_p_c).sum(axis=1)
        # state info gain
        p_oo = q_ss @ p_o # prior
        joint = q_ss[...,None] * p_o
        q_oo = joint / joint.sum( axis=2, keepdims=True ) # conditional
        d_oo = (q_oo * (jax_safelog( q_oo ) - jax_safelog( q_ss )[...,None])).sum( axis=2 ) # KL
        info_gain = (d_oo * p_oo).sum(axis=(1, 2)) # sum over o and t
        nefe = pragmatic + info_gain
        return q, nefe
    
    nefe = jax.lax.scan(batch_nefe, init=q, xs=πs)[1] # scan over batches
    # action selection
    π = πs.reshape(-1, πs.shape[-1])[jnp.argmax(nefe)]
    # propagate belief through time
    q = q @ p_t[:,π[0],:]
    return q, π

jit_step_batched_fun = jax.jit(jax_step_batched_fun, static_argnums=(0,))

class MinimalAgentJax:
    """ Minimal agent performing exact inference in fully discrete POMDPs"""
    def __init__(self, 
                 env,
                 target_state, 
                 k=2, 
                 num_batches=1):
        self.env = env
        self.target_state = target_state
        self.k = k
        self.num_batches = num_batches
        
        self.p_t = jnp.asarray(env.p_s1_given_s_a)
        self.p_o = jnp.asarray(env.p_o_given_s)
        print(f'Enumerating {self.env.a_N**k:,} candidate policies of length {k}')
        self.πs = np.stack(np.meshgrid(*[np.arange(self.env.a_N) for _ in range(k)])).T.reshape(-1, k)
        self.πs = jax.device_put(self.πs)
        # slow naive implementation
        #self.πs = jnp.asarray([x for x in itertools.product( range(self.env.a_N), repeat=self.k )])
        
    def reset(self):
        # initialize state preference
        q_star = np.eye(self.env.s_N)[self.target_state] \
                 + 1/(self.env.s_N*5) * np.ones(self.env.s_N)
        self.q_star = q_star / q_star.sum()
        self.log_p_c = jnp.log( self.q_star )
        # initialize state prior as uniform
        self.q = jnp.asarray(np.ones(self.env.s_N) / self.env.s_N )
    
    def step(self, o):
        params = {
            'num_actions': self.env.a_N,
            'o': o,
            'q': self.q,
            'p_o': self.p_o,
            'p_t': self.p_t,
            'log_p_c': self.log_p_c,
            'πs': self.πs.reshape(self.num_batches, -1, self.k),
        }
        self.q, π = jit_step_batched_fun(**params)
        return π[0]
    
    
def jax_fully_observed_batched_fun(num_actions, q, πs, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value in fully-observed environment.
    Args:
        num_actions (int): number of unique actions. Jit assumes this is static across calls.
        q (jnp.ndarray): one-hot encoding of the observed state
        πs (jnp.ndarray): 3D tensor of actions [batch, policy, action_at_time_t].
        p_t (jnp.ndarray): 3D tensor of transition dynamics [s0, a, s1].
        log_p_c (jnp.ndarray): 1D tensor of normalized state preferences. One element per environment state.
    """
    # policy rollout: coalesced access to transition dynamics speeds up vectorised rollouts across policies
    def step(q, a):
        # iterate over values of a
        def step_a(i, q):
            is_a = (a==i)
            # weighted average between q and q' with binary weight
            q = (1-is_a) * q + is_a * (q @ p_t[:,i,:]) # replace with q + is_a (q @ p_t[:,i,:] - q)
            return q
        
        q = jax.lax.fori_loop(0, num_actions, step_a, q)
        return q, q
    
    qs_π = lambda q, π: jax.lax.scan(step, init=q, xs=π)[1]
    qs_batch = jax.vmap(qs_π, in_axes=(None, 0), out_axes=(0)) # carry, output
    pragmatic = jax.lax.scan(lambda q, πs: (q, (qs_batch(q, πs) @ log_p_c).sum(axis=-1)), init=q, xs=πs)[1] # scan over batches
    π = πs.reshape(-1, πs.shape[-1])[jnp.argmax(pragmatic)]
    return π

jit_fully_observed_batched_fun = jax.jit(jax_fully_observed_batched_fun, static_argnums=(0,))

class FullyObservedAgentJax:
    """ Minimal agent performing exact inference in fully discrete POMDPs"""
    def __init__(self, 
                 env,
                 target_state, 
                 k=2, 
                 num_batches=1):
        self.env = env
        self.target_state = target_state
        self.k = k
        self.num_batches = num_batches
        
        self.p_t = jnp.asarray(env.p_s1_given_s_a)
        print(f'Enumerating {self.env.a_N**k:,} candidate policies of length {k}')
        self.πs = np.stack(np.meshgrid(*[np.arange(self.env.a_N) for _ in range(k)])).T.reshape(-1, k)
        self.πs = jax.device_put(self.πs)
        
    def reset(self):
        # initialize state preference
        q_star = np.eye(self.env.s_N)[self.target_state] \
                 + 1/(self.env.s_N*5) * np.ones(self.env.s_N)
        q_star = q_star / q_star.sum()
        self.log_p_c = jnp.log( q_star )
        # initialize state prior as uniform
    
    def step(self, o):
        params = {
            'q': jax.nn.one_hot(o, self.env.s_N),
            'p_t': self.p_t,
            'log_p_c': self.log_p_c,
            # batch policies
            'πs': self.πs.reshape(self.num_batches, -1, self.k), 
            'num_actions': self.env.a_N
        }
        return jit_fully_observed_batched_fun(**params)