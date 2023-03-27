import itertools

import numpy as np
import torch

def softmax(x):
  e = np.exp(x - x.max())
  return e / e.sum()

def kl(a, b):
    """ Discrete KL-divergence """
    return (a * (np.log(a) - np.log(b))).sum()

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
        
        self.πs = np.asarray( 
          [x for x in itertools.product( range(self.env.a_N), repeat=self.k )]
        )
        
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
      
  
class MinimalAgentVI:
    
    def __init__(self, 
                 env,
                 target_state, 
                 k=2, # planning horizon
                 use_info_gain=True, # score actions by info gain
                 use_pragmatic_value=True, # score actions by pragmatic value
                 select_max_pi=False, # sample plan (False), select max negEFE (True).
                 n_steps_o=20, # optimization steps after new observation
                 n_steps_a=20, # optimization steps after new action
                 lr_o=4., # learning rate of optimization after new observation
                 lr_a=4.): # learning rate of optimization after new action)
        
        self.env = env
        self.target_state = target_state
        self.k = k
        self.use_info_gain = use_info_gain
        self.use_pragmatic_value = use_pragmatic_value
        self.select_max_pi = select_max_pi
        self.n_steps_o = n_steps_o
        self.n_steps_a = n_steps_a
        self.lr_a = lr_a
        self.lr_o = lr_o
        
    def reset(self):
        # initialize state preference
        self.b_star = np.eye(self.env.s_N)[self.target_state] * 10
        self.log_p_c = np.log(softmax(self.b_star))
        # initialize state prior as uniform
        self.b = np.zeros(self.env.s_N)
        
    def step(self, o, debug=False):
        if debug:
            return self._step_debug(o)
        
        self.b = self._update_belief(theta_prev=self.b, o=int(o))
        a = select_action(theta_start=self.b)[0] # pop first action of selected plan
        self.b = self._update_belief_a(theta_prev=self.b, a=a)
        return a
    
    def _step_debug(self, o):
        self.b, ll_o = self._update_belief(theta_prev=self.b, 
                                           o=int(o), debug=True)
        a, p_a, _, _ = self._select_action(theta_start=self.b, debug=True)
        a = a[0]
        self.b, ll_a = self._update_belief_a(theta_prev=self.b, a=a, debug=True)
        return a, ll_o, ll_a, p_a
    
    def _update_belief_a(self, theta_prev, a, debug=False):
        # prior assumed to be expressed as parameters of the softmax (logits)
        theta = torch.tensor(theta_prev)
        q = torch.nn.Softmax(dim=0)(theta)

        # this is the prior for the distribution at time t
        q1 = torch.matmul(q, torch.tensor(self.env.p_s1_given_s_a[:,a,:]))

        # initialize parameters of updated belief to uniform
        theta1 = torch.zeros_like(theta, requires_grad=True)
        loss = torch.nn.CrossEntropyLoss() # expects logits and target distribution.
        optimizer = torch.optim.SGD([theta1], lr=self.lr_a)
        if debug:
            ll = np.zeros(self.n_steps_a)

        for i in range(self.n_steps_a):
            l = loss(theta1, q1)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if debug:
                ll[i] = l.detach().numpy()

        theta1 = theta1.detach().numpy()
        if debug:
            return theta1, ll

        return theta1
    
    def _update_belief(self, theta_prev, o, debug=False):
        theta = torch.tensor(theta_prev)

        # make p(s) from b
        q = torch.nn.Softmax(dim=0)
        p = torch.tensor(self.env.p_o_given_s[:,o]) * q(theta) # p(o|s)p(s)
        log_p = torch.log(p)

        # initialize updated belief with current belief
        theta1 = torch.tensor(theta_prev, requires_grad=True)

        # estimate loss
        def forward():
            q1 = q(theta1)
            # free energy: KL[ q(s) || p(s, o) ]
            fe = torch.sum(q1 * (torch.log(q1) - log_p))
            return fe

        optimizer = torch.optim.SGD([theta1], lr=self.lr_o)
        ll = np.zeros(self.n_steps_o)
        for i in range(self.n_steps_o):
            l = forward()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if debug:
                ll[i] = l.detach().numpy()

        theta1 = theta1.detach().numpy()
        if debug:
            return theta1, ll

        return theta1

    def _select_action(self, theta_start,
                       debug=False): # return plans, p of selecting each, and marginal p of actions
        # sampling
        #n_plans = 32
        #plans = np.random.choice(n_actions, size=(n_plans, k), replace=True).tolist()
        # genrate all plans
        plans = [ list(x) for x in itertools.product(range(self.env.a_N), repeat=self.k)]
        # evaluate negative expected free energy of all plans
        nefes = []
        for pi in plans:
            step_nefes = self._rollout_step(theta_start, pi)
            nefe = np.array(step_nefes).mean() # expected value over steps
            nefes.append(nefe)

        # compute probability of following each plan
        p_pi = softmax(np.array(nefes)).tolist()
        if self.select_max_pi:
            a = plans[np.argmax(nefes)]
        else:
            a = plans[np.random.choice(len(plans), p=p_pi)]

        if debug:
            # compute marginal action probabilities
            p_a = np.zeros(self.env.a_N)
            for p, pi in zip(p_pi, plans):
                p_a[pi[0]] += p

            return a, p_a, plans, p_pi

        return a

    def _rollout_step(self, theta, pi):
        if pi == []:
            return []

        a, pi_rest = pi[0], pi[1:]
        # Where will I be after taking action a?
        theta1 = self._update_belief_a(theta, a=a) 
        q = softmax(theta1)
        # Do I like being there?
        pragmatic = np.dot(q, self.log_p_c)
        # What might I observe after taking action a? (marginalize p(o, s) over s)
        p_o = np.dot(q, self.env.p_o_given_s)
        # Do I learn about s from by observing o?
        # enumerate/ sample observations, update belief and estimate info gain
        q_o = [softmax(self._update_belief(theta1, o=i)) for i in range(p_o.shape[0])]
        d_o = [kl(q_o_i, q) for q_o_i in q_o] # info gain for each observation
        info_gain = np.dot(p_o, d_o) # expected value of info gain
        # negative expected free energy for this timestep
        nefe = self.use_pragmatic_value * pragmatic + \
               self.use_info_gain * info_gain
        # nefe for remainder of policy rollout
        nefe_rest = self._rollout_step(theta1, pi_rest)
        # concatenate expected free energy across future time steps
        return [nefe] + nefe_rest