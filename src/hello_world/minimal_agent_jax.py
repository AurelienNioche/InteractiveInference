import subprocess as sp
import numpy as np

import jax
import jax.numpy as jnp

print(f"JAX devices: {jax.devices()}")
print(f"JAX device type: {jax.devices()[0].device_kind}")

# check GPU status
def get_gpu_memory():
    print(sp.check_output("nvidia-smi").decode('ascii'))

def jax_safelog(x):
  return jnp.log( jnp.maximum(x, 1e-16 ))

def jax_step_naive_fun(o, q, πs, p_o, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value and state information gain.
    
    This implementation is left here for illustrative purposes only. Optimised functions for production are below.
    It won't run directly within agents, because several changes have been made to the input formt
        - policies are not batches of policies with actions one-hot encoded
        - first input argument in num_actions
    """
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

def jax_step_batched_fun(o, q, πs, p_o, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value and state information gain."""
    # update belief from new observation
    joint = q * p_o[:,o]
    q = joint / joint.sum()
    
    # policy rollout: 
    # one-hot action encoding ensures coalesced access to transition dynamics, 
    # speeding up vectorised rollouts across policies
    def step(q, a):
        q = a @ (q @ p_t)
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
    π = πs.reshape( (-1,) + πs.shape[2:] )[jnp.argmax(nefe)] # perform policy selection by squeezing batch and policy dimensions
    π = jnp.argmax(π, axis=-1)
    # propagate belief through time
    q = q @ p_t[π[0],:,:]
    return q, π

jit_step_batched_fun = jax.jit(jax_step_batched_fun)

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
        
        self.p_t = jnp.asarray(env.p_s1_given_s_a.swapaxes(0,1))
        self.p_o = jnp.asarray(env.p_o_given_s)
        print(f'Enumerating {self.env.a_N**k:,} candidate policies of length {k}')
        #batched policies with one-hot actions
        πs = np.stack(np.meshgrid(*[np.arange(self.env.a_N) for _ in range(k)])).T.reshape(self.num_batches, -1, k)
        self.πs = jax.nn.one_hot(jax.device_put(πs), self.env.a_N, dtype=int)
        
        # slow naive implementation
        #self.πs = jnp.asarray([x for x in itertools.product( range(self.env.a_N), repeat=self.k )])
        
    def reset(self):
        # initialize state preference
        q_star = np.eye(self.env.s_N)[self.target_state] * 10
        self.log_p_c = q_star
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
        self.q, π = jit_step_batched_fun(**params) if use_jit else jax_step_batched_fun(**params)
        return π[0]
    
    
def jax_fully_observed_batched_fun(q, πs, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value in fully-observed environment.
    Args:
        num_actions (int): number of unique actions. Jit assumes this is static across calls.
        q (jnp.ndarray): one-hot encoding of the observed state
        πs (jnp.ndarray): 4D tensor of one-hot actions [batch, policy, timestep, one_hot_action].
        p_t (jnp.ndarray): 3D tensor of transition dynamics [a, s0, s1].
        log_p_c (jnp.ndarray): 1D tensor of normalized state preferences. One element per environment state.
    """
    # policy rollout: 
    # one-hot action encoding ensures coalesced access to transition dynamics, 
    # speeding up vectorised rollouts across policies
    def step(q, a):
        q = a @ (q @ p_t)
        return q, q
    
    qs_π = lambda q, π: jax.lax.scan(step, init=q, xs=π)[1] # carry, output
    qs_batch = jax.vmap(qs_π, in_axes=(None, 0), out_axes=(0)) # parallel rollout of policy batch
    # batch computation of negative free energy (here, only pragmatic value)
    def nefe_batch(q, πs):
        pragmatic = (qs_batch(q, πs) @ log_p_c).sum(axis=-1)
        return q, pragmatic
    
    nefe = jax.lax.scan(nefe_batch, init=q, xs=πs)[1] # scan over batches
    π = πs.reshape( (-1,) + πs.shape[2:] )[jnp.argmax(nefe)] # perform policy selection by squeezing first two dimensions
    π = jnp.argmax(π, axis=-1)
    return π

jit_fully_observed_batched_fun = jax.jit(jax_fully_observed_batched_fun)

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
        
        self.p_t = jnp.asarray(env.p_s1_given_s_a.swapaxes(0,1))
        print(f'Enumerating {self.env.a_N**k:,} candidate policies of length {k}')
        #batched policies
        πs = np.stack(np.meshgrid(*[np.arange(self.env.a_N) for _ in range(k)])).T.reshape(self.num_batches, -1, k)
        # one-hot encoded actions
        self.πs = jax.nn.one_hot(jax.device_put(πs), self.env.a_N, dtype=int)
        
    def reset(self):
        # initialize state preference
        q_star = np.eye(self.env.s_N)[self.target_state] * 10
        self.log_p_c = q_star
    
    def step(self, o):
        params = {
            'q': jax.nn.one_hot(o, self.env.s_N),
            'p_t': self.p_t,
            'log_p_c': self.log_p_c,
            'πs': self.πs, 
        }
        return jit_fully_observed_batched_fun(**params)
    
    
def jax_fully_observed_nefe(π_logits, q, p_t, log_p_c):
    """ Deterministic policy selection with pragmatic value in fully-observed environment.
    Args:
        num_actions (int): number of unique actions. Jit assumes this is static across calls.
        q (jnp.ndarray): one-hot encoding of the observed state
        πs (jnp.ndarray): 4D tensor of one-hot actions [batch, policy, timestep, one_hot_action].
        p_t (jnp.ndarray): 3D tensor of transition dynamics [a, s0, s1].
        log_p_c (jnp.ndarray): 1D tensor of normalized state preferences. One element per environment state.
    """

    def step(q, a):
        q = a @ (q @ p_t)
        return q, q
    
    # calculate nefe
    π = jax.nn.softmax(π_logits)
    qs_π = jax.lax.scan(step, init=q, xs=π)[1] # carry, output
    pragmatic = (qs_π @ log_p_c).sum(axis=-1)
    nefe = pragmatic
    return nefe

def jax_update(π_logits, q, p_t, log_p_c, lr):
    eval_fun = jax.value_and_grad(jax_fully_observed_nefe)
    v, g = eval_fun(π_logits, q, p_t, log_p_c)
    π_logits = π_logits + lr * g
    return v, π_logits

class FullyObservedGradAgentJax:
    """ Minimal agent performing exact inference in fully discrete POMDPs"""
    def __init__(self, 
                 env,
                 target_state, 
                 k=2, 
                 num_grad_steps=100, 
                 learning_rate=2.):
        self.env = env
        self.target_state = target_state
        self.k = k
        self.p_t = jnp.asarray(env.p_s1_given_s_a.swapaxes(0,1))
        self.num_gradient_steps = num_grad_steps
        self.learning_rate = learning_rate
        
    def reset(self):
        # initialize state preference
        q_star = np.eye(self.env.s_N)[self.target_state] * 10
        self.log_p_c = jnp.log( jnp.exp( q_star ))
    
    def step(self, o, debug=True):
        π_logits = jnp.zeros( (self.k, self.env.a_N) ) # initialize uniform
        params = {
            'q': jax.nn.one_hot(o, self.env.s_N),
            'p_t': self.p_t,
            'log_p_c': self.log_p_c,
            'lr': self.learning_rate
        }
        vv = []
        for _ in range(self.num_gradient_steps):
            v, π_logits = jax.jit(jax_update)(π_logits, **params)
            vv.append(v)
        
        π = π_logits.argmax(axis=-1)
        return π, vv, π_logits if debug else π