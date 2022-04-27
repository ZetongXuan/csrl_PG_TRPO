# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:54:11 2022

@author: Zetong
"""

"""Performs the TRPO algorithm and returns the Policy

Parameters
----------
start : int
    The start state of the MDP.
    
T : int
    The episode length.

K : int 
    The number of episodes.
    
Returns
-------
theta: array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions) 
    The policy learned.
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:41:17 2022

@author: Zetong 

Apply Policy Gradient method to the product MDP
with value esitmation as baseline
"""


from csrl.mdp import GridMDP
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import softmax
    
def softmax_state(theta,state,csrl):
    Action_set = csrl.A[state]  
    # temp = theta[state][:len(Action_set)] / np.linalg.norm(theta[state][:len(Action_set)])
    temp = theta[state][:len(Action_set)]
    # Policy_A_theta = np.exp(temp) / np.sum(np.exp(temp))
    Policy_A_theta = softmax(temp)
    return Policy_A_theta

def derivate_softmax(theta,state,csrl,action,Policy_A_theta):
    Action_set = csrl.A[state]
    Pi_A_theta = Policy_A_theta[Action_set.index(action)]
    Grad_Pi = np.zeros([len(Action_set),1])
    for i in range(len(Grad_Pi)):
        if i == Action_set.index(action):
            Grad_Pi[i] = Pi_A_theta*(1-Pi_A_theta)
        else:
            Grad_Pi[i] = -Pi_A_theta*Policy_A_theta[i]
    return Grad_Pi

def cg(A, b): 
    if ~np.any(b):
        x = b
        return x
    cg_iters = 10
    residual_tol=1e-10   
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)  
    rdotr = np.dot(r,r)  
    for i in range(cg_iters):   
        z = A @ p
        v = rdotr / p.dot(z)
        x = x + v*p
        r = r - v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p  
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

def transition_matrix():
    # policy_dist =  np.zeros(theta.shape)
    dim = csrl.shape[:-1][0]*csrl.shape[:-1][1]*csrl.shape[:-1][2]*csrl.shape[:-1][3]
    P_pi = np.zeros([dim,dim])
    index = 0
    R_pi = np.zeros([dim,1])
    for i,q,r,c in csrl.states():
        state = (i,q,r,c)
        R_pi[index] = csrl.reward[state]
        A_set   = csrl.A[state]
        # softmax, from theta to probability
        Policy_s_theta = softmax_state(theta,state,csrl)
        states_array = []    
        probs_array = []
        for action in A_set:
            Pi_A_theta = Policy_s_theta[A_set.index(action)]
            if action < len(csrl.mdp.A): # MDP actions
                q_ = oa.delta[q][csrl.mdp.label[r,c]]  # OA transition
                mdp_states, probs = csrl.mdp.get_transition_prob((r,c),csrl.mdp.A[action])  # MDP transition
                states = [(i,q_,)+s for s in mdp_states]
                probs = np.array(probs) * Pi_A_theta
                probs = probs.tolist()
            else:  # epsilon-actions
                states, probs = ([(i,action-len(csrl.mdp.A),r,c)], [1.])
                probs = np.array(probs) * Pi_A_theta
                probs = probs.tolist()
            states_array = states_array +  states
            probs_array = probs_array + probs
    
        for k in range(len(states_array)):
            s_plus = states_array[k]
            s_plus = s_plus[1]*csrl.shape[:-1][2]*csrl.shape[:-1][3] +  s_plus[2]*csrl.shape[:-1][3] +  s_plus[3]
            # s_plus = int(s_plus)
            P_pi[s_plus,index] = probs_array[k] + P_pi[s_plus,index]            
        index = index +1
    return P_pi, R_pi

def one_trajectroy(initial_state,csrl,theta,length): 
    state   = initial_state
    reward  = csrl.reward[state]
    state_hist  = [state]
    reward_hist = [reward]
    gamma_hist  = [csrl.discountB if reward else csrl.discount]
    action_hist = []
    for t in range(length):
        A_set = csrl.A[state]
        # softmax, from theta to probability
        # Policy, action distributation given state and theta
        action = A_set[np.random.choice(len(A_set),p= softmax_state(theta,state,csrl) )]
        states, probs = csrl.transition_probs[state][action]
        state = states[np.random.choice(len(states),p=probs)]
        reward = csrl.reward[state]
        gamma = csrl.discountB if reward else csrl.discount
        state_hist.append(state)
        reward_hist.append(reward)
        gamma_hist.append(gamma)
        action_hist.append(action)
    state_hist  = state_hist[:-1]
    reward_hist = reward_hist[:-1]
    gamma_hist  = gamma_hist[:-1]
    G_t = reward_hist[-1]
    G_t_hist = [G_t]
    for t in range(length-2,-1,-1):
        G_t = reward_hist[t] + gamma_hist[t] * G_t
        G_t_hist.append(G_t)
    G_t_hist.reverse()
    return state_hist, action_hist, G_t_hist, gamma_hist
#%%
# LTL Specification
ltl = '(F G a | F G b) & G !c'

# Translate the LTL formula to an LDBA
oa = OmegaAutomaton(ltl)
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])

# MDP Description
shape = (5,4)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
['E',  'E',  'E',  'E'],
['E',  'E',  'E',  'T'],
['B',  'E',  'E',  'E'],
['T',  'E',  'T',  'E'],
['E',  'E',  'E',  'E']
])

# Labels of the states
label = np.array([
[(),       (),     ('c',),()],
[(),       (),     ('a',),('b',)],
[(),       (),     ('c',),()],
[('b',),   (),     ('a',),()],
[(),       ('c',), (),    ('c',)]
],dtype=object)
lcmap={
    ('a',):'lightgreen',
    ('b',):'lightgreen',
    ('c',):'pink'
}
grid_mdp = GridMDP(shape=shape,structure=structure,label=label,lcmap=lcmap,figsize=5)  # Use figsize=4 for smaller figures
# grid_mdp.plot()

csrl = ControlSynthesis(grid_mdp,oa) # Product MDP
#%%
T = 100
K = 15000

theta = np.ones(csrl.shape)
V = np.zeros(csrl.shape[:-1])
V_hist = np.zeros([K+1] + list(csrl.shape[:-1]) )
Policy_hist = np.zeros([K+1] + list(csrl.shape[:-1]) )

theta_hist = np.zeros( [K+1] + list(csrl.shape) )
theta_hist[0] = theta
eig_max_hist = np.zeros([K])
eig_min_hist = np.zeros([K])
di_norm_hist = np.zeros([K])

for k in tqdm(range(K)):
    # k_th episode
    # initial state
    state   = (csrl.shape[0]-1,csrl.oa.q0)+(csrl.mdp.random_state())
    state_hist, action_hist, G_t_hist, gamma_hist = one_trajectroy(state,csrl,theta,T+200)
# =============================================================================
# use 1 trajectory to generate policy gradient and update parameter once, 
    # alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
    alpha = 1
    PG = np.zeros(csrl.shape)
    Grad_theta_Pi = np.zeros(csrl.shape)
    Grad_V = np.zeros(csrl.shape[:-1])
    for t in range(T):
        state   = state_hist[t]
        action  = action_hist[t]
        A_set   = csrl.A[state]
        # softmax, from theta to probability
        Policy_s_theta = softmax_state(theta,state,csrl)
        Pi_A_theta = Policy_s_theta[A_set.index(action)]
        # derivate of log softmax, pi w.r.t. theta
        Grad_Pi = derivate_softmax(theta,state,csrl,action,Policy_s_theta) / Pi_A_theta
        

        PG_state = np.prod(gamma_hist[0:t:1]) * (G_t_hist[t] - V[state]) *  Grad_Pi
        PG[state][0:len(PG_state):1] += PG_state.flatten()
        Grad_V[state] = Grad_V[state] + (G_t_hist[t]- V[state])
        
        Grad_log_Pi_state = np.prod(gamma_hist[0:t:1]) * Grad_Pi
        Grad_theta_Pi[state][0:len(PG_state):1] += Grad_log_Pi_state.flatten()
        
    FIM = Grad_theta_Pi.flatten().reshape(-1,1)  @ Grad_theta_Pi.flatten().reshape(-1,1).T

    x =  np.linalg.pinv(FIM) @ PG.flatten()
    delta = 0.05
    x_THx = x.reshape(-1,1).T @ FIM @ x.reshape(-1,1)
    di = np.sqrt( 2*delta / (x_THx if x_THx else 1) )   * x

    di = di.reshape(theta.shape)
    # di_norm_hist[k] = np.linalg.norm(di)
    # theta = theta   +   alpha * di / T
    theta = theta   +   0.5*di
    V =     V       +   0.05*Grad_V
        
    V_hist[k+1] = V
    theta_hist[k+1] = theta
# =============================================================================
#%% from theta to policy
policy = np.zeros(csrl.shape[:-1])
for i,q,r,c in csrl.states():
    state = (i,q,r,c)
    # temp = theta[state][0:len(csrl.A[state])]
    temp = theta[state][csrl.A[state]]
    policy[state] = np.argmax(temp)
    policy = policy.astype(int)
print('policy')
print(policy[0,0])
#%%
# solve linear equation to get value funciton based on current policy
P_pi, R_pi = transition_matrix()
dim = csrl.shape[:-1][0]*csrl.shape[:-1][1]*csrl.shape[:-1][2]*csrl.shape[:-1][3]
V_pi = np.linalg.inv(np.eye(dim)-0.99*P_pi.T) @ R_pi
V_pi_table = np.zeros(csrl.shape[:-1])
index = 0 
for i,q,r,c in csrl.states():
    V_pi_table[i,q,r,c] = V_pi[index]
    index = index +1
V_pi_table = np.round(V_pi_table,2)
print( np.around( (np.sum(V_hist[round(k/10*9):k:1],axis=0)/(k/10))[0,0],decimals = 2 ))
print(V_pi_table[0,0])
#%%
plt.figure()
# plt.hold(True)
plt.plot(range(k),V_hist[0:k:1,0,0,0,0])
plt.plot(range(k),V_hist[0:k:1,0,0,1,2])
plt.plot(range(k),V_hist[0:k:1,0,0,1,3])
plt.plot(range(k),V_hist[0:k:1,0,0,3,1])
#%%
plt.figure()
plt.plot(range(k),theta_hist[0:k:1,0,0,1,0])
#%%
# plt.figure()
# plt.plot(range(k),eig_max_hist[0:k:1])
# plt.plot(range(k),eig_min_hist[0:k:1])

# plt.figure()
# plt.plot(range(k),di_norm_hist[0:k:1])