"""
LELME2150 - Thermal cycles
Homework 1 - Basic cycles

Signature of the function for the Brayton cycle

"""

#
#===IMPORT PACKAGES============================================================
#

from CoolProp.CoolProp import PropsSI
import numpy as np
import scipy as sc
# Please do not add any other packages

#
#===BRAYTON CYCLE - TO BE IMPLEMENTED==========================================
#

class Brayton_cycle(object):
    """
    Class for the simulation of Brayton cycles

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
        Create a Brayton cycle object.
        """
        p_1,T_1,p_3,T_3         = inputs
        self.p_1,self.T_1       = p_1, T_1                  # pressure [Pa] and temperature [K] at the compressor inlet
        self.p_3,self.T_3       = p_3, T_3                  # pressure [Pa] and temperature [K] at the turbine    inlet
        
        self.eta_pi_c           = parameters['eta_pi_c']    # compressor polytropic efficiency     [-]
        self.eta_pi_t           = parameters['eta_pi_t']    # turbine    polytropic efficiency     [-]
        self.eta_mec_c          = parameters['eta_mec_c']   # compressor mechanical efficiency     [-]
        self.eta_mec_t          = parameters['eta_mec_t']   # turbine    mechanical efficiency     [-]
        self.k_cc               = parameters['k_cc']        # pressure loss coeff. in comb. chamb. [-]
        self.fluid              = parameters['fluid']       # fluid species (e.g. ['N2','O2'])
        self.fluid_prop         = parameters['fluid_prop']  # species proportions (e.g. [0.79, 0.21])
        self.R             = PropsSI('GAS_CONSTANT','O2') # gas constant

        self.fluid_string = 'SRK::'
        for i in range(len(self.fluid)):
            self.fluid_string += self.fluid[i] + '[' + str(self.fluid_prop[i]) + ']'
            if i < len(self.fluid)-1:
                self.fluid_string += '&'
    
        self.M_m_air      = PropsSI('MOLARMASS',self.fluid_string) # molar mass of air
        self.R_star       = self.R/self.M_m_air # specific gas constant of air

        self.integration_points = 20 # number of points for the numerical integration of Cp and Cp/T

    def evaluate(self):
        """
        This is the main method of the Brayton cycle class.
        It evaluates the different thermodynamic quantities at each state of 
        the Brayton cycle, as well as some KPI's.
        """

        
        # >>>>>             <<<<< #    
        # Replace with your model #
        # >>>>>             <<<<< # 
        
        # Combustion chamber -----------------------------
        self.p_2 = self.p_3/self.k_cc
        # Compressor -------------------------------------
        gamma_c_in = PropsSI('CPMASS','P',self.p_1,'T',self.T_1,self.fluid_string)/PropsSI('CVMASS','P',self.p_1,'T',self.T_1,self.fluid_string)
        self.T_2 = self.T_1*(self.p_2/self.p_1)**((gamma_c_in-1)/(gamma_c_in*self.eta_pi_c))
        nb_iter = 0
        while True:
            previous_T_2 = self.T_2
            T_vals = np.linspace(self.T_1,self.T_2,self.integration_points)
            gamma_c_moy = sc.integrate.simpson(PropsSI('CPMASS','P',self.p_1,'T',T_vals,self.fluid_string)/PropsSI('CVMASS','P',self.p_1,'T',T_vals,self.fluid_string), x=T_vals)/(self.T_2-self.T_1)
            self.T_2 = self.T_1*(self.p_2/self.p_1)**((gamma_c_moy-1)/(gamma_c_moy*self.eta_pi_c))
            nb_iter += 1
            if abs(self.T_2 - previous_T_2) < 1e-6 or nb_iter > 10:
                break
            
        # Air ambiant ------------------------------------
        self.p_4 = self.p_1
        # Turbine ----------------------------------------
        gamma_t_in = PropsSI('CPMASS','P',self.p_3,'T',self.T_3,self.fluid_string)/PropsSI('CVMASS','P',self.p_3,'T',self.T_3,self.fluid_string) # on suppose que le gamma de la turbine va rester constant et égal à celui de l'entrée de la turbine
        self.T_4 = self.T_3*(self.p_4/self.p_3)**(((gamma_t_in-1)*self.eta_pi_t)/(gamma_t_in))
        nb_iter = 0
        while True:
            previous_T_4 = self.T_4
            T_vals = np.linspace(self.T_3,self.T_4,self.integration_points)
            gamma_t_moy = sc.integrate.simpson(PropsSI('CPMASS','P',self.p_3,'T',T_vals,self.fluid_string)/PropsSI('CVMASS','P',self.p_3,'T',T_vals,self.fluid_string), x=T_vals)/(self.T_4-self.T_3)
            self.T_4 = self.T_3*(self.p_4/self.p_3)**(((gamma_t_moy-1)*self.eta_pi_t)/(gamma_t_moy))
            nb_iter += 1
            if abs(self.T_4 - previous_T_4) < 1e-6 or nb_iter > 10:
                break

        # Thermodynamic properties (h-h1,s-s1) -----------
            # enthalpy
        self.h_1 = PropsSI('H','P',self.p_1,'T',self.T_1,self.fluid_string)
        self.h_2 = PropsSI('H','P',self.p_2,'T',self.T_2,self.fluid_string)
        self.h_3 = PropsSI('H','P',self.p_3,'T',self.T_3,self.fluid_string)
        self.h_4 = PropsSI('H','P',self.p_4,'T',self.T_4,self.fluid_string)
            # entropy
        self.s_1 = PropsSI('S','P',self.p_1,'T',self.T_1,self.fluid_string)
        self.s_2 = PropsSI('S','P',self.p_2,'T',self.T_2,self.fluid_string)
        self.s_3 = PropsSI('S','P',self.p_3,'T',self.T_3,self.fluid_string)
        self.s_4 = PropsSI('S','P',self.p_4,'T',self.T_4,self.fluid_string)

        
        # Cycle efficiency - do not modify
        self.w_comp = (self.h_2-self.h_1)/self.eta_mec_c
        self.q_comb = (self.h_3-self.h_2)
        self.w_turb = (self.h_3-self.h_4)*self.eta_mec_t
        self.eta_en = (self.w_turb-self.w_comp)/self.q_comb        
        # Final outputs - do not modify
        self.p = (self.p_1, self.p_2, self.p_3, self.p_4)
        self.T = (self.T_1, self.T_2, self.T_3, self.T_4)
        self.s = (self.s_1, self.s_2, self.s_3, self.s_4)
        self.h = (self.h_1, self.h_2, self.h_3, self.h_4)
        self.states = (self.p,self.T,self.s,self.h)
        
if __name__ == '__main__':
        # print(PropsSI('CPMASS','P',1e+5,'T',300,'O2'))
        # print(PropsSI('GAS_CONSTANT','HEOS::O2[0.9]&N2[0.1]'))
        # R = PropsSI('GAS_CONSTANT','O2')
        # M_m_air = 0.21*PropsSI('MOLARMASS','O2') + 0.79*PropsSI('MOLARMASS','N2')
        # print(M_m_air, PropsSI('MOLARMASS','HEOS::O2[0.21]&N2[0.79]'))
        # R_star = R/M_m_air
        # print(R_star)
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
        pass