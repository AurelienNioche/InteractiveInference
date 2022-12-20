import numpy as np

# agent
"""
- the agent has an internal brain state b
- it receives a sensory state o in each timestep
- it chooses an action a to minimise variational free energy
- it updates its brain state to b' based on b, a, and s

"""
def softmax(x):
  e = np.exp(x - x.max())
  return e / e.sum()

def KL(a, b):
  """ Discrete KL divergence."""
  return np.dot(a, (np.log(a) - np.log(b)))

def get_b_star(s_star=0, s_N=16):
  b = np.zeros(s_N)
  b[s_star] = 10
  return b

class MinimalAgent(object):

  def __init__(self,
               p_s1_given_s_a, # true environment transition probability
               p_o_given_s, # true environment emission probability
               b_star, # logits of desired state distribution
               a_N=2, # number of discrete actions
               b_N=16, # number of internal states (tabular representation of p(b))
               ):
    
    # environment dynamics
    self.p_s1_given_s_a = p_s1_given_s_a
    self.p_o_given_s = p_o_given_s
    self.a_N = a_N
    
    # belief state
    self.b_N = b_N # number of belief states
    self.b_star = b_star # desired distribution over belief states
    self.b_t = None # current belief state (undefined before reset)

  def reset(self):
    self.b_t = np.zeros(self.b_N) # uniform belieft at start

  def act(self, o):
    min_fe = None
    argmin_fe = None

    # evaluate policies by evaluating single next action
    # - more generally, we evaluate trajectories of actions (pi: a_0, ..., a_tau)
    # - normally, we'd pick actions by sampling from softmax(G_pi)
    # - here, we take action with maximum g
    for a in range(self.a_N):
      # Free Energy is KL between p* and E_s~q [p(s')]
      # Note: we use action indices to represent actions (not {-1, 1})
      fe = self.free_energy(self.b_star, o, a)
      if (min_fe is None) or (fe < min_fe):
        min_fe = fe
        argmin_fe = a

      
    return argmin_fe

  @staticmethod
  def q(b):
    """ Variational distribution of environment state s given belief state b.
        p(s|b)

      (model_encoding, variational_density)
    """
    return softmax(b)

  @classmethod
  def dq(self, b):
    """ Derivative of the variational distribution.
     (model_encoding_derivative)
    """
    q = self.q(b)
    # Softmax derivative
    return np.diag(q) - np.outer(q, q)

  def generative_density(self, b, o, a):
    """
    Next state prediction from generative model.
    Here, the generative model is equal to the true environment dynamics.
    (generative_density)

    P(s', o | b, a) = Sum_over_s(P(s' | a, s) * P(o | s) * P(s | b))
    s' only depends on a and s, o only depends on s, and s only depends on b.

    Agent's prediction of next state probability given belief state and action
    (calculated separately for both sensory states).
    """

    # generative model of the next state p(s1, o | b, a)
    # todo: adapt to return joint for both observations
    p_o_s_given_b = self.q(self.b_t) * self.p_o_given_s[:,o] # joint prob p(o, s| b)
    p_s1_o_given_b_a = np.dot(p_o_s_given_b, self.p_s1_given_s_a[:,a,:])
    return p_s1_o_given_b_a

  def free_energy(self, b_star, o, a):
    # estimate of expected free energy, used for action selection
    q = self.q(b_star) # where I want to be
    p = self.generative_density(self.b_t, o, a=a) # where I get to taking action a
    return KL(q, p)

  def update_state(self, o, a, n_steps, lr=1.0):
    # internal belief state at time t+1 can be initialised
    # a) uniformly (expressing minimal knowledge about the future)
    # b) biased towards the current state (assuming small changes)
    # c) by updating current belief according to current world model
    #    this assumes that we know the inverse q^-1(b|s) which, in general, we don't
    b_prime = np.copy(self.b_t) # (b), alternatively np.zeros(self.b_N) (a)

    # posterior joint of next state and last observation given  last action 
    # and last belief state. This is constant across update iterations
    p = self.generative_density(self.b_t, o, a)
    #plt.plot(p, label="$p(s', o | b, a)$")

    for i in range(n_steps):
      q = self.q(b_prime)
      # KL(q, p)
      #F = np.dot(q, (np.log(q) - np.log(p))
      
      # free energy gradient wrt belief state
      dq = self.dq(b_prime)
      Y = 1 + (np.log(q) - np.log(p))
      db = np.dot(dq, Y)

      b_prime -= lr * db

    self.b_t = b_prime
