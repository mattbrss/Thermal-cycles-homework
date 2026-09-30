"""
LELME2150 - Thermal cycles
Homework 1 - Basic cycles

Test code for your functions

Code execution can take a few seconds...

"""
#
#===IMPORT PACKAGES============================================================
#

import os
import sys
HW1 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HW1,'corrections'))

import numpy as np
import matplotlib.pyplot as plt
from brayton_cycle_group_02 import Brayton_cycle
from rankine_cycle_group_02 import Rankine_cycle

#
#===PLOT FUNCTIONS - IMPLEMENTED FOR YOU=======================================
#

size = 50

def plot_eta_GT(p_1,T_1,p_3,T_3,eta_pi,eta_mec_c,eta_mec_t,k_cc):
    p_3_test, eta_p_3_test = np.linspace(2e+5,100e+5,size),      np.zeros(size)    
    T_3_test, eta_T_3_test = np.linspace(873.15,1673.15,size),   np.zeros(size)    
    eta_pi_test,  eta_eta_pi_test = np.linspace(0.7,1,size),     np.zeros(size)   
    
    for i in range(size):
        
        inputs = p_1,T_1,p_3_test[i],T_3
        params =  {'eta_pi_c':  eta_pi,
                   'eta_pi_t':  eta_pi,
                   'eta_mec_c': eta_mec_c,
                   'eta_mec_t': eta_mec_t,
                   'k_cc':      k_cc,
                   'fluid':     ['O2','N2'],
                   'fluid_prop':[0.21,0.79]}
        my_GT = Brayton_cycle(inputs,params)
        my_GT.evaluate()
        eta_p_3_test[i] = my_GT.eta_en
        
        inputs = p_1,T_1,p_3,T_3_test[i]
        params =  {'eta_pi_c':  eta_pi,
                   'eta_pi_t':  eta_pi,
                   'eta_mec_c': eta_mec_c,
                   'eta_mec_t': eta_mec_t,
                   'k_cc':      k_cc,
                   'fluid':     ['O2','N2'],
                   'fluid_prop':[0.21,0.79]}
        my_GT = Brayton_cycle(inputs,params)
        my_GT.evaluate()
        eta_T_3_test[i] = my_GT.eta_en
        
        inputs = p_1,T_1,p_3,T_3
        params =  {'eta_pi_c':  eta_pi_test[i],
                   'eta_pi_t':  eta_pi_test[i],
                   'eta_mec_c': eta_mec_c,
                   'eta_mec_t': eta_mec_t,
                   'k_cc':      k_cc,
                   'fluid':     ['O2','N2'],
                   'fluid_prop':[0.21,0.79]}
        my_GT = Brayton_cycle(inputs,params)
        my_GT.evaluate()
        eta_eta_pi_test[i] = my_GT.eta_en    

    fig = plt.figure(figsize=[15,5])
    axs = fig.subplots(1,3)
    axs[0].plot(p_3_test*1e-5,eta_p_3_test,'-',color='tab:red',linewidth=2)
    axs[0].plot(p_3*1e-5,eta_en,'ok',linewidth=2)
    print('eta_en_brayton : '+str(eta_en))
    axs[0].set_xlabel('Compressor outlet p [bar]',fontsize=13)
    axs[0].set_xticks([0,20,40,60,80,100])
    axs[0].set_ylabel('GT energy efficiency [-]',fontsize=13)
    axs[0].set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])    
    axs[1].plot(T_3_test-273.15,eta_T_3_test,'-',color='tab:red',linewidth=2)
    axs[1].plot(T_3-273.15,eta_en,'ok',linewidth=2)
    axs[1].set_xlabel('Combustor outlet T [°C]',fontsize=13)
    axs[1].set_xticks([600,800,1000,1200,1400])
    axs[1].set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
    axs[2].plot(eta_pi_test,eta_eta_pi_test,'-',color='tab:red',linewidth=2)
    axs[2].plot(eta_pi,eta_en,'ok',linewidth=2)
    axs[2].set_xlabel('Polytropic efficiency [-]',fontsize=13)
    axs[2].set_xticks([0.7,0.8,0.9,1.0])  
    axs[2].set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
    fig.tight_layout()
    plt.show()
    return fig

def plot_eta_ST(T_1,p_3,T_3,LHV,P_el,eta_mec_t,eta_is_t,eta_pump,eta_gen,k_gen):
    p_3_test, eta_p_3_test          = np.linspace(60e+5,160e+5,size),    np.zeros(size)    
    T_3_test, eta_T_3_test          = np.linspace(623.15,1023.15,size),  np.zeros(size)    
    eta_is_test,  eta_eta_is_test   = np.linspace(0.7,1,size),           np.zeros(size)    
    
    for i in range(size):
        
        inputs = T_1,p_3_test[i],T_3,P_el,LHV
        params = {'eta_pump':   eta_pump,
                  'eta_is_t':   eta_is_t,
                  'eta_mec_t':  eta_mec_t,
                  'eta_gen':    eta_gen,
                  'k_gen':      k_gen,
                  'fluid':      'H2O'}
        my_ST = Rankine_cycle(inputs,params)
        my_ST.evaluate()
        eta_p_3_test[i] = my_ST.eta_en
        
        inputs = T_1,p_3,T_3_test[i],P_el,LHV
        params = {'eta_pump':   eta_pump,
                  'eta_is_t':   eta_is_t,
                  'eta_mec_t':  eta_mec_t,
                  'eta_gen':    eta_gen,
                  'k_gen':      k_gen,
                  'fluid':      'H2O'}
        my_ST = Rankine_cycle(inputs,params)
        my_ST.evaluate()
        eta_T_3_test[i] = my_ST.eta_en

        inputs = T_1,p_3,T_3,P_el,LHV
        params = {'eta_pump':   eta_pump,
                  'eta_is_t':   eta_is_test[i],
                  'eta_mec_t':  eta_mec_t,
                  'eta_gen':    eta_gen,
                  'k_gen':      k_gen,
                  'fluid':      'H2O'}
        my_ST = Rankine_cycle(inputs,params)
        my_ST.evaluate()
        eta_eta_is_test[i] = my_ST.eta_en
        
    fig = plt.figure(figsize=[15,5])
    axs = fig.subplots(1,3)
    axs[0].plot(p_3_test*1e-5,eta_p_3_test,'-',color='tab:red',linewidth=2)
    axs[0].plot(p_3*1e-5,eta_en,'ok',linewidth=20)
    print('eta_en_rankine : '+str(eta_en))
    axs[0].set_xlabel('Boiler outlet p [bar]',fontsize=13)
    axs[0].set_xticks([60, 80, 100, 120, 140, 160])
    axs[0].set_ylabel('ST energy efficiency [-]',fontsize=13)
    axs[0].set_yticks([0.1,0.2,0.3,0.4])    
    axs[1].plot(T_3_test-273.15,eta_T_3_test,'-',color='tab:red',linewidth=2)
    axs[1].plot(T_3-273.15,eta_en,'ok',linewidth=2)
    axs[1].set_xlabel('Boiler outlet T [°C]',fontsize=13)
    axs[1].set_xticks([350,450,550,650,750])
    axs[1].set_yticks([0.1,0.2,0.3,0.4])
    axs[2].plot(eta_is_test,eta_eta_is_test,'-',color='tab:red',linewidth=2)
    axs[2].plot(eta_is_t,eta_en,'ok',linewidth=2)
    axs[2].set_xlabel('Isentropic efficiency [-]',fontsize=13)
    axs[2].set_xticks([0.7,0.8,0.9,1.0])  
    axs[2].set_yticks([0.1,0.2,0.3,0.4])
    fig.tight_layout()
    plt.show()
    return fig

#
#===BRAYTON CYCLE==============================================================
#

p_1, T_1 = 1e+5, 293.15 # [Pa], [K]
p_3, T_3 = 17.8e+5, 1273.15 # [Pa], [K]
eta_pi = 0.90 # [-]
eta_mec_c, eta_mec_t = 0.98, 0.98 # [-], [-]
k_cc = 1

inputs = p_1,T_1,p_3,T_3
params =  {'eta_pi_c':  eta_pi,
           'eta_pi_t':  eta_pi,
           'eta_mec_c': eta_mec_c,
           'eta_mec_t': eta_mec_t,
           'k_cc':      k_cc,
           'fluid':     ['O2','N2'],
           'fluid_prop':[0.21,0.79]}
my_GT = Brayton_cycle(inputs,params)
my_GT.evaluate()
eta_en = my_GT.eta_en

fig_GT = plot_eta_GT(p_1,T_1,p_3,T_3,eta_pi,eta_mec_c,eta_mec_t,k_cc)

#
#===RANKINE CYCLE==============================================================
#

T_1 = 303.15 # [K]
p_3, T_3 = 100e+5, 813.15 # [Pa], [K]
eta_is_t = 0.85 # [-]
eta_mec_t = 0.985 # [-]
eta_pump = 0.8 # [-]
eta_gen = 0.9 #0.6 [-]
k_gen = 0.91
LHV = 25e+6 # [J/kg]
P_el = 95e+6 # [W]

inputs = T_1,p_3,T_3,P_el,LHV
params = {'eta_pump':   eta_pump,
          'eta_is_t':   eta_is_t,
          'eta_mec_t':  eta_mec_t,
          'eta_gen':    eta_gen,
          'k_gen':      k_gen,
          'fluid':      'H2O'}
my_ST = Rankine_cycle(inputs,params)
my_ST.evaluate()
eta_en = my_ST.eta_en

fig_ST = plot_eta_ST(T_1,p_3,T_3,LHV,P_el,eta_mec_t,eta_is_t,eta_pump,eta_gen,k_gen)


# %%
