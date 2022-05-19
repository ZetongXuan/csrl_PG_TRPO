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
# import scipy.special as sc
from scipy.special import softmax
import dill
from numba import jit

def softmax_state(theta,state,csrl):
    Action_set = csrl.A[state]  
    # temp = theta[state][:len(Action_set)] / np.linalg.norm(theta[state][:len(Action_set)])
    temp = theta[state][:len(Action_set)]
    # Policy_A_theta = softmax(temp)
    Policy_A_theta = np.exp(temp) / np.sum(np.exp(temp))
    return Policy_A_theta

def derivate_softmax(theta,state,csrl,action,Policy_A_theta):
    Action_set = csrl.A[state]
    P_a = Policy_A_theta[Action_set.index(action)]
    Grad_Pi = np.zeros([len(Action_set),1])
    for i in range(len(Grad_Pi)):
        if i == Action_set.index(action):
            Grad_Pi[i] = P_a*(1-P_a)
        else:
            Grad_Pi[i] = -P_a * Policy_A_theta[i]
    return Grad_Pi

# @jit(nopython=False)
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
        

        # action = A_set[np.argmax(softmax_state(theta,state,csrl) )]
        
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
ltl = 'F G a'

# Translate the LTL formula to an LDBA
oa = OmegaAutomaton(ltl)
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])

# MDP Description
shape = (2,2)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
['E',  'E'],
['E',  'T']
])

# Labels of the states
label = np.array([
[(),       ()],
[(),       ('a',)]
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
# csrl.discount = 0.99
# csrl.discountB = 0.9
T = 100
K = 10000
np.random.seed(2)

theta = np.zeros(csrl.shape)
theta = np.random.random(csrl.shape)
theta_hist = np.zeros( [K+1] + list(csrl.shape) )
theta_hist[0] = theta
theta_hist = np.zeros( [K+1] + list(csrl.shape) )
V = np.zeros(csrl.shape[:-1])
V_hist = np.zeros([K+1] + list(csrl.shape[:-1]) )
#%%
# dill.dump_session('300K_PG_base.pkl')
# dill.load_session('300K_PG_base.pkl')
#%%
for k in tqdm(range(K)):
    # k_th episode
# =============================================================================
# simulate and record a trajectroy,
    # initial state    
    state   = (csrl.shape[0]-1,csrl.oa.q0)+(csrl.mdp.random_state())  
    state_hist, action_hist, G_t_hist, gamma_hist = one_trajectroy(state,csrl,theta,T+200)
# =============================================================================
# use 1 trajectory to generate policy gradient and update parameter once, 
    # alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
    alpha = 0.05

    for t3 in range(T):

        state   = state_hist[t3]
        if not (state in state_hist[0:t3]) :            
            action  = action_hist[t3]
            A_set   = csrl.A[state]
            # softmax, from theta to probability
            Policy_s = softmax_state(theta,state,csrl)
            P_a = Policy_s[A_set.index(action)]
            # derivate of log softmax, pi w.r.t. theta
            Grad_Pi = derivate_softmax(theta,state,csrl,action,Policy_s) / P_a
            
            PG = np.zeros(theta.shape)
            Grad_V = np.zeros(V.shape)
            
            # PG_state = np.prod(gamma_hist[0:t3:1])* (G_t_hist[t3] + np.prod(gamma_hist[t3:-1:1]) * V[state_hist[-1]]- V[state]) * Grad_Pi 
            PG_state = np.prod(gamma_hist[0:t3:1])* (G_t_hist[t3]-V[state_hist[t3]]) * Grad_Pi 
            PG[state][0:len(PG_state):1] += PG_state.flatten()
            Grad_V[state] = G_t_hist[t3] - V[state]    
            
            theta = theta + 0.1*PG
            V     = V     + alpha * Grad_V
        
        # PG_state = np.prod(gamma_hist[0:t3:1]) * (G_t_hist[t3] - V[state]) * Grad_Pi / P_a
        
    #     PG_state = np.prod(gamma_hist[0:t3:1]) * (G_t_hist[t3] + V[state_hist[-1]] - V[state]) * Grad_Pi 
    #     PG[state][0:len(PG_state):1] += PG_state.flatten()
    #     Grad_V[state] = Grad_V[state] + G_t_hist[t3] - V[state]       
    # theta = theta   + PG / T
    # V     = V       + Grad_V/T
    
    V_hist[k+1] = V
    theta_hist[k+1] = theta
# =============================================================================
#%% from theta to policy
policy = np.zeros(csrl.shape[:-1])
for i,q,r,c in csrl.states():
    state = (i,q,r,c)
    # temp = theta[state][0:len(csrl.A[state])]
    temp = theta[state][0:len(csrl.A[state])]
    policy[state] = csrl.A[state][np.argmax(temp)]
    policy = policy.astype(int)
print(policy[0,0])

print( np.around( (np.sum(V_hist[round(k/10*9):k:1],axis=0)/(k/10))[0,0],decimals = 2 ))
#%%
# solve linear equation to get value funciton based on current policy
# P_pi, R_pi = transition_matrix()
# dim = csrl.shape[:-1][0]*csrl.shape[:-1][1]*csrl.shape[:-1][2]*csrl.shape[:-1][3]
# V_pi = np.linalg.inv(np.eye(dim)-0.9*P_pi.T) @ R_pi
# V_pi_table = np.zeros(csrl.shape[:-1])
# index = 0 
# for i,q,r,c in csrl.states():
#     V_pi_table[i,q,r,c] = V_pi[index]
#     index = index +1
# V_pi_table = np.round(V_pi_table,2)
# # print(V_pi_table[0,0])
# #$$ PR averaging

# print(V_pi_table[0,0])


#%%
plt.figure()
# plt.hold(True)
plt.plot(range(k),V_hist[0:k:1,0,0,0,0])
plt.plot(range(k),V_hist[0:k:1,0,0,0,1])
plt.plot(range(k),V_hist[0:k:1,0,0,1,0])
plt.plot(range(k),V_hist[0:k:1,0,0,1,1])

#%%
plt.figure()
plt.plot(range(k),theta_hist[0:k:1,0,0,0,0])
plt.figure()
plt.plot(range(k),theta_hist[0:k:1,0,0,0,1])
plt.figure()
plt.plot(range(k),theta_hist[0:k:1,0,0,1,0])
plt.figure()
plt.plot(range(k),theta_hist[0:k:1,0,0,1,1])