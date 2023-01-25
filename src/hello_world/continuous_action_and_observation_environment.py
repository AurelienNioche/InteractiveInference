import numpy as np
from scipy.stats import beta

# environment
class ContinuousAOEnv(object):
  """ Wrap-around 1D state space with single food source.
  
  The probability of sensing food at locations near the food source decays 
  exponentially with increasing distance.
  
  state (int): 1 of N discrete locations in 1D space.
  observation (float): proportion of times food detected in finite sample.
  actions(float): [-2, 2] intention to move left or right.
  """
  def __init__(self, 
               N = 16, # how many discrete locations can the agent reside in
               s_0 = 0, # where does the agent start each episode?
               s_food = 0, # where is the food?
               sigma_move = 0.75, # Gaussian stdev around continuous move
               o_sample_size=10, # observation Beta distribution parameter.
               a_lims = [-2, 2], # maximum step in either direction.
               p_o_max = 0.9, # maximum probability of sensing food
               o_decay = 0.2 # decay rate of observing distant food source
               ):
    
    self.o_decay = o_decay
    self.var_move = sigma_move**2
    self.o_sample_size = o_sample_size
    self.p_o_max = p_o_max
    self.s_0 = s_0
    self.s_food = s_food
    self.s_N = N
    self.o_N = 2 # {False, True} indicating whether food has been found
    self.a_lims = a_lims
    """
    environment dynamics are governed by two probability distributions
    1. state transition probability p(s'|s, a)
    2. emission/ observation probability p(o|s)
    
    With continuous-valued actions, we can nolonger represent (1.) with a 
    single conditional probability table. However, we can generate one table of
    size |S| x |S| for each continuous action value.
    """
    self.d_s = self._signed_state_distances()
    # self.p_s1_given_s_a(a=a) returns p[s, s1] for given a; slice of Matrix B
    
    """
    We pre-compute the conditional emission random variables (2.) here so agents 
    can access the true dynamics if required.
    """
    self.p_o_given_s = self.emission_probability() # Matrix A
    
    self.s_t = None # state at current timestep

  def _signed_state_distances(self):
    s = np.arange(self.s_N)
    other, this = np.meshgrid(s, s)
    d = other - this
    d1 = other - this + self.s_N
    d2 = other - this - self.s_N
    d[np.abs(d) > np.abs(d1)] = d1[np.abs(d) > np.abs(d1)]
    d[np.abs(d) > np.abs(d2)] = d2[np.abs(d) > np.abs(d2)]
    return d
  
  def _p_a_discrete_given_a(self, a):
    # probability distribution of a discrete action (step) given a continuous
    # action intent.
    a = np.clip(a, self.a_lims[0], self.a_lims[1])
    a_discrete = np.arange(2*self.s_N-1) - self.s_N + 1
    p_a = np.exp(-0.5 * (a_discrete-a)**2 / self.var_move)
    p_a[a_discrete > self.a_lims[1]] = 0
    p_a[a_discrete < self.a_lims[0]] = 0
    p_a = p_a/p_a.sum()
    return a_discrete, p_a
  
  def p_s1_given_s_a(self, a):
    """ computes transition probability p(s'| s, a) for specific a
    
    Note: this is provided for convenience in the agent; it is not used within
    the environment simulation.

    Returns:
    p[s, s1] of size (s_N, s_N)
    """
    a_d, p_a = self._p_a_discrete_given_a(a=a)
    return p_a[self.d_s - a_d[0]]

  def emission_probability(self):
    """ initialises conditional random variables p(o|s). 
    
    Returns:
    p[s] of size (s_N) with one scipy.stats.rv_continuous per state
    """
    s = np.arange(self.s_N)
    # distance from food source
    d = np.minimum(np.abs(s - self.s_food), 
                   np.minimum(
                   np.abs(s - self.s_N - self.s_food), 
                   np.abs(s + self.s_N - self.s_food)))
  
    # exponentially decaying concentration ~ probability of detection
    mean = self.p_o_max * np.exp(-self.o_decay * d)
    # continuous relaxation: proportion of food detected in finite sample
    sample_size = self.o_sample_size
    return np.array([beta(a=m*sample_size, b=(1-m)*sample_size) for m in mean])

  def reset(self):
    self.s_t = self.s_0
    return self.sample_o()

  def step(self, a):
    if (self.s_t is None):
      print("Warning: reset environment before first action.")
      self.reset()
      
    a_discrete = self.sample_a(a)
    self.s_t = (self.s_t + a_discrete) % self.s_N
    return self.sample_o()

  def sample_o(self):
    return self.p_o_given_s[self.s_t].rvs()
  
  def sample_a(self, a):
    a_d, p_a = self._p_a_discrete_given_a(a=a)
    return np.random.choice(a_d, p=p_a)