import numpy as np
from scipy.stats import beta

# environment
class ContinuousObservationEnv(object):
  """ Wrap-around 1D state space with single food source.
  
  The probability of sensing food at locations near the food source decays 
  exponentially with increasing distance.
  
  state (int): 1 of N discrete locations in 1D space.
  observation (float): proportion of times food detected in finite sample.
  actions(int): {-1, 1} intention to move left or right.
  """
  def __init__(self, 
               N = 16, # how many discrete locations can the agent reside in
               s_0 = 0, # where does the agent start each episode?
               s_food = 0, # where is the food?
               p_move = 0.75, # execute intent with p, else don't move.
               o_sample_size=10, # observation Beta distribution parameter.
               p_o_max = 0.9, # maximum probability of sensing food
               o_decay = 0.2 # decay rate of observing distant food source
               ):
    
    self.o_decay = o_decay
    self.p_move = p_move
    self.o_sample_size = o_sample_size
    self.p_o_max = p_o_max
    self.s_0 = s_0
    self.s_food = s_food
    self.s_N = N
    self.a_N = 2 # {0, 1} to move left/ right in wrap-around 1D state-space
    """
    environment dynamics are governed by two probability distributions
    1. state transition probability p(s'|s, a)
    2. emission/ observation probability p(o|s)
    although we only need to be able to sample from these distributions to 
    implement the environment, we pre-compute the full conditional probability
    table (1.) and conditional emission random variables (2.) here so agents 
    can access the true dynamics if required.
    """
    self.p_s1_given_s_a = self.transition_dynamics() # Matrix B
    self.p_o_given_s = self.emission_probability() # Matrix A
    self.s_t = None # state at current timestep


  def transition_dynamics(self):
    """ computes transition probability p(s'| s, a) 
    
    Returns:
    p[s, a, s1] of size (s_N, a_N, s_N)
    """

    p = np.zeros((self.s_N, self.a_N, self.s_N))
    p[:,0,:] = self.p_move * np.roll(np.identity(self.s_N), -1, axis=1) \
              + (1-self.p_move) * np.identity(self.s_N)
    p[:,1,:] = self.p_move * np.roll(np.identity(self.s_N), 1, axis=1) \
              + (1-self.p_move) * np.identity(self.s_N)
    return p

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
    return [beta(a=m*sample_size, b=(1-m)*sample_size) for m in mean]

  def reset(self):
    self.s_t = self.s_0
    return self.sample_o()

  def step(self, a):
    if (self.s_t is None):
      print("Warning: reset environment before first action.")
      self.reset()

    if (a not in [0, 1]):
      print("Warning: only permitted actions are [0, 1].")

    # convert action index to action
    a = [-1,1][a]

    if np.random.random() < self.p_move:
      self.s_t = (self.s_t + a) % self.s_N
    return self.sample_o()

  def sample_o(self):
    return self.p_o_given_s[self.s_t].rvs()