import numpy as np

def softmax(x):
  e = np.exp(x - x.max())
  return e / e.sum()

def np_safelog(x):
  return np.log( np.maximum(x, 1e-16) )

class MinimalAgent:
    """ Minimal agent performing exact inference in fully discrete POMDPs"""
    def __init__(self, 
                 env,
                 target_state, 
                 k=2, # planning horizon
                 β=1, # Bolzmann inverse temperature for policy sampling
                 use_info_gain=True, # score actions by info gain
                 use_pragmatic_value=True, # score actions by pragmatic value
                 select_max_π=False): # sample plan (False), select max negEFE (True).
        self.env = env
        self.target_state = target_state
        self.k = k
        self.β = β
        self.use_info_gain = use_info_gain
        self.use_pragmatic_value = use_pragmatic_value
        self.select_max_π = select_max_π
        print(f'Enumerating {self.env.a_N**k:,} candidate policies of length {k}')
        self.πs = np.stack(np.meshgrid(*[np.arange(self.env.a_N) for _ in range(k)])).T.reshape(-1, k)
        
    def reset(self):
        # initialize state preference
        q_star = np.eye(self.env.s_N)[self.target_state] \
                 + 0.01 * np.ones(self.env.s_N)
        self.q_star = q_star / q_star.sum()
        self.log_p_c = np.log( self.q_star )
        # initialize state prior as uniform
        self.q = np.ones(self.env.s_N) / self.env.s_N
        
    def step(self, o):
        # update belief based on observation
        joint = self.q * self.env.p_o_given_s[:,int(o)]
        self.q = joint / joint.sum()
        π = self._select_π()
        # propagate belief through time assuming we take action a
        self.q = self.q @ self.env.p_s1_given_s_a[:,π[0],:]
        return π[0]
      
    @staticmethod
    def _qs_π(p, q_start, π):
        """ propagate q through env following all steps in pi. """
        num_t = π.shape[0]
        q_ss = np.empty(shape= π.shape + q_start.shape)
        q = q_start
        for i, a in zip( range(num_t), π ):
          q = q @ p[:,a,:]
          q_ss[i] = q
        return q_ss
      
    def _select_π(self, debug=False):
      # rollout
      q_ss = np.stack([self._qs_π(self.env.p_s1_given_s_a, self.q, π) for π in self.πs]) # policies x T x states
      # pragmatic value
      pragmatic = (q_ss @ self.log_p_c).sum(axis=1)
      # state info gain
      p_oo = q_ss @ self.env.p_o_given_s # prior
      joint = q_ss[...,None] * self.env.p_o_given_s
      q_oo = joint / joint.sum( axis=2, keepdims=True ) # conditional
      d_oo = (q_oo * (np_safelog( q_oo ) - np_safelog( q_ss )[...,None])).sum( axis=2 ) # KL
      info_gain = (d_oo * p_oo).sum(axis=(1, 2)) # sum over o and t
      #action selection
      nefe = self.use_pragmatic_value* pragmatic + self.use_info_gain * info_gain
      p_πs = softmax(self.β * nefe)
      if self.select_max_π:
          return self.πs[ np.argmax(nefe) ]
      else:    
          return self.πs[ np.random.choice( num_πs, p=p_πs ) ]
      