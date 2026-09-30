"""
LELME2150 - Thermal cycles
Homework 1 - Basic cycles

Signature of the function for the Rankine cycle

"""

#
#===IMPORT PACKAGES============================================================
#

from CoolProp.CoolProp import PropsSI
import numpy as np
import scipy as sc
# Please do not add any other packages

#
#===RANKINE CYCLE - TO BE IMPLEMENTED==========================================
#

class Rankine_cycle(object):
    """
    Class for the simulation of the Rankine cycle

    Parameters
    ----------
    'inputs':
        -> Tuple with the model inputs (SI units = K, Pa).
    'parameters':
        -> Dictionary with the model parameters.
        
    Methods
    -------
    'evaluate':
        -> Compute the cycle and evaluate some KPI's.
    """
    
    def __init__(self,inputs,parameters):
        """
        Create a Rankine cycle object.
        """
        T_1,p_3,T_3,P_el,LHV = inputs
        self.T_1                = T_1                       # temperature [K] at the pump inlet
        self.p_3,self.T_3       = p_3, T_3                  # pressure [Pa] and temperature [K] at the turbine inlet
        self.P_el, self.LHV     = P_el, LHV                 # demanded power [W] and fuel LHV [J/kg]
        
        self.eta_pump           = parameters['eta_pump']    # pump internal efficiency          [-]
        self.eta_is_t           = parameters['eta_is_t']    # turbine isentropic efficiency     [-]
        self.eta_mec_t          = parameters['eta_mec_t']   # turbine mechanical efficiency     [-]
        self.eta_gen            = parameters['eta_gen']     # steam generator efficiency        [-]
        self.k_gen              = parameters['k_gen']       # pressure losses coeff. in s. gen. [-]
        self.fluid              = parameters['fluid']       # fluid (e.g. 'H2O')
        
    def evaluate(self):
        """
        This is the main method of the Rankine cycle class.
        It evaluates the different thermodynamic quantities at each state of 
        the Rankine cycle, as well as some KPI's.
        """
    
        # >>>>>             <<<<< #    
        # Replace with your model #
        # >>>>>             <<<<< # 

        #code
        Tsat = self.T_1+3
        psat = PropsSI('P','T',Tsat,'Q',0,self.fluid)
        self.p_1 = psat
        self.h_1 = PropsSI('H','T',self.T_1,'P',self.p_1,self.fluid)
        self.s_1 = PropsSI('S','T',self.T_1,'P',self.p_1,self.fluid)
        self.x_1 = PropsSI('Q','T',self.T_1,'P',self.p_1,self.fluid)
        self.h_3 = PropsSI('H','T',self.T_3,'P',self.p_3,self.fluid)
        self.s_3 = PropsSI('S','T',self.T_3,'P',self.p_3,self.fluid)
        self.x_3 = PropsSI('Q','T',self.T_3,'P',self.p_3,self.fluid)
        #as we know the kgen, for the state2:
        self.p_2 = self.p_3/self.k_gen
        s_2s = self.s_1
        h_2s = PropsSI('H','P',self.p_2,'S',s_2s,self.fluid)
        self.h_2 = self.h_1 + (h_2s-self.h_1)/self.eta_pump
        self.T_2 = PropsSI('T','P',self.p_2,'H',self.h_2,self.fluid)
        self.s_2 = PropsSI('S','P',self.p_2,'H',self.h_2,self.fluid)
        self.x_2 = PropsSI('Q','P',self.p_2,'H',self.h_2,self.fluid)
        #and stage 4 with isobaric condensor
        s_4s = self.s_3
        self.p_4 = self.p_1
        h_4s = PropsSI('H','P',self.p_4,'S',s_4s,self.fluid)
        self.h_4 = self.h_3 - (self.h_3-h_4s)*self.eta_is_t
        self.T_4 = PropsSI('T','P',self.p_4,'H',self.h_4,self.fluid)
        self.s_4 = PropsSI('S','P',self.p_4,'H',self.h_4,self.fluid)
        self.x_4 = PropsSI('Q','P',self.p_4,'H',self.h_4,self.fluid)

        #we search the pressure that maximizes the efficiency for the propane
        #(in the other code, basic_cycles_test, we just put propane instead of H2O, and T3=373.15K)
        #we found approximatively p(etamax) = 40 bars, with an order magnitude of O(38 to 42 bars)





        
        # Cycle efficiency - do not modify
        self.W_pump = (self.h_2-self.h_1)
        self.Q_comb = (self.h_3-self.h_2)/self.eta_gen
        self.W_turb = (self.h_3-self.h_4)*self.eta_mec_t
        self.eta_en = (self.W_turb-self.W_pump)/self.Q_comb
        self.dot_m_f= self.P_el/self.LHV/self.eta_en
        # Final outputs - do not modify
        self.p = (self.p_1, self.p_2, self.p_3, self.p_4)
        self.T = (self.T_1, self.T_2, self.T_3, self.T_4)
        self.s = (self.s_1, self.s_2, self.s_3, self.s_4)
        self.h = (self.h_1, self.h_2, self.h_3, self.h_4)
        self.x = (self.x_1, self.x_2, self.x_3, self.x_4)
        self.states = (self.p,self.T,self.s,self.h,self.x)