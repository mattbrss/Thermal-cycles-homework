"""
LELME2150 - Thermal cycles
Homework 2 - Gas turbine

Signature of the function for the gas turbine

"""

#
#===IMPORT PACKAGES============================================================
#
import CoolProp.CoolProp as CP
from scipy.integrate import quad, simpson
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
#
#===BRAYTON CYCLE - TO BE IMPLEMENTED==========================================
#

class gas_turbine(object):
    """
    Class for the simulation of gas turbines

    Parameters
    ----------
    'inputs':
        -> Tuple with the model inputs (SI units = K, Pa).
    'parameters':
        -> Dictionary with the model parameters.
    'display':
        -> Boolean to plot the graphs.
        
    Methods
    -------
    'evaluate':
        -> Compute the cycle
    """
    
    def __init__(self,inputs,parameters,display):
        """
        Create a gas turbine object.
        """
        p_1,T_1,P_e             = inputs
        self.p_1                = p_1                       # [Pa] compressor supply pressure
        self.T_1                = T_1                       # [K]  compressor supply temperature
        self.P_e                = P_e                       # [W]  gas turbine power output
        self.T_3                = parameters['T_3']         # [K]  turbine inlet temperature 
        self.r_c                = parameters['r_c']         # [-]  compression ratio
        self.eta_pi_c           = parameters['eta_pi_c']    # [-]  polytropic efficiency of compressor
        self.eta_pi_t           = parameters['eta_pi_t']    # [-]  polytropic efficiency of turbine
        self.k_cc               = parameters['k_cc']        # [-]  pressure   losses coefficient in combustion chamber
        self.k_mec              = parameters['k_mec']       # [-]  mechanical losses coefficient
        self.air                = parameters['air']         # air  species
        self.air_prop           = parameters['air_prop']    # molar fractions of air species
        self.alkane             = parameters['alkane']      # alkane composition ['c','h']
        self.x=self.alkane[0]; self.y=self.alkane[1]
        self.x_N2_0 = self.air_prop[self.air.index('N2')] # [-] molar fraction of N2 in the inlet air
        self.x_O2_0 = self.air_prop[self.air.index('O2')] # [-] molar fraction of O2 in the inlet air

        self.display            = display                   # if True, make the plots
        self.R                  = CP.PropsSI('GAS_CONSTANT','O2') # gas constantant
        self.p_0_air            = 100000                    # [Pa] reference pressure for air exergy
        self.T_0                = 288.15                    # [K]  reference temperature for air exergy
        self.p_0_H20            = CP.PropsSI('P','T',288.15,'Q',0,'H2O')  # [Pa] reference pressure for water exergy

        self.air_string = 'SRK::'
        for i in range(len(self.air)):
            self.air_string += self.air[i] + '[' + str(self.air_prop[i]) + ']'
            if i < len(self.air)-1:
                self.air_string += '&'

        self.M_m_air      = CP.PropsSI('MOLARMASS',self.air_string) # molar mass of air
        self.R_star       = self.R/self.M_m_air # specific gas constant of air
        self.integration_points = 20 # number of points for the numerical integration of Cp

    def n_tot(self,excess_air):
        """
        Compute the total number of moles in the combustion products
        """

        n_tot = (self.x+(self.y/4))*(excess_air-1) + self.x + (self.y/2) +(self.x_N2_0/self.x_O2_0)*(self.x+(self.y/4))*excess_air # [-] total number of moles in the combustion products
        return n_tot
    def x_O2(self,excess_air):
        """
        Compute the molar fraction of O2 in the combustion products
        """
        x_O2 = (self.x+(self.y/4))*(excess_air-1)/self.n_tot(excess_air) # [-] molar fraction of O2 in the combustion products
        return x_O2
    def x_N2(self,excess_air):
        """
        Compute the molar fraction of N2 in the combustion products
        """
        x_N2 = (self.x_N2_0/self.x_O2_0)*(self.x+(self.y/4))*excess_air/self.n_tot(excess_air) # [-] molar fraction of N2 in the combustion products
        return x_N2
    def x_H2O(self,excess_air):
        """
        Compute the molar fraction of H2O in the combustion products
        """
        x_H2O = (self.y/2)/self.n_tot(excess_air) # [-] molar fraction of H2O in the combustion products
        return x_H2O
    def x_CO2(self,excess_air):
        """
        Compute the molar fraction of CO2 in the combustion products
        """
        x_CO2 = self.x/self.n_tot(excess_air) # [-] molar fraction of CO2 in the combustion products
        return x_CO2
    
    def evaluate(self):
        """
        This is the main method of the gas turbine class.
        It evaluates the different thermodynamic quantities at each state of 
        the cycle, as well as some KPI's.
        """
        # State 0 (reference state) -------------------------------------------
        self.h_0_air = CP.PropsSI('H','P',self.p_0_air,'T',self.T_0,self.air_string) # [J/kg] enthalpy at state 0
        self.s_0_air = CP.PropsSI('S','P',self.p_0_air,'T',self.T_0,self.air_string) # [J/kg/K] entropy at state 0
        # State 1 -------------------------------------------------------------
        self.h_1 = CP.PropsSI('H','P',self.p_1,'T',self.T_1,self.air_string) # [J/kg] enthalpy at state 1
        self.s_1 = CP.PropsSI('S','P',self.p_1,'T',self.T_1,self.air_string) # [J/kg/K] entropy at state 1
        self.e_1 = (self.h_1-self.h_0_air) - self.T_0*(self.s_1-self.s_0_air) # [J/kg] exergy at state 1
        # State 2 -------------------------------------------------------------
        self.p_2 = self.r_c*self.p_1
        gamma_c_in = CP.PropsSI('CPMASS','P',self.p_1,'T',self.T_1,self.air_string)/CP.PropsSI('CVMASS','P',self.p_1,'T',self.T_1,self.air_string)
            # Fixed point iteration to find T2
        self.T_2 = self.T_1*(self.p_2/self.p_1)**((gamma_c_in-1)/(gamma_c_in*self.eta_pi_c)) # First guess for T2
        nb_iter = 0
        while True:
            previous_T_2 = self.T_2
            T_vals = np.linspace(self.T_1,self.T_2,self.integration_points)
            gamma_c_moy = simpson(CP.PropsSI('CPMASS','P',self.p_1,'T',T_vals,self.air_string)/CP.PropsSI('CVMASS','P',self.p_1,'T',T_vals,self.air_string), x=T_vals)/(self.T_2-self.T_1)
            self.T_2 = self.T_1*(self.p_2/self.p_1)**((gamma_c_moy-1)/(gamma_c_moy*self.eta_pi_c))
            nb_iter += 1
            if abs(self.T_2 - previous_T_2) < 1e-6 or nb_iter > 10:
                break
            # ------------------------------------
        self.h_2 = CP.PropsSI('H','P',self.p_2,'T',self.T_2,self.air_string) # [J/kg] enthalpy at state 2
        self.s_2 = CP.PropsSI('S','P',self.p_2,'T',self.T_2,self.air_string) # [J/kg/K] entropy at state 2
        self.e_2 = (self.h_2-self.h_0_air) - self.T_0*(self.s_2-self.s_0_air) # [J/kg] exergy at state 2
        # State 3 -------------------------------------------------------------
        self.p_3 = self.p_2*self.k_cc
            # Fixed point iteration to find h3
        self.lamda = 1 # First guess for lambda
        self.h_3 = self.x_O2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"O2") + self.x_N2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"N2") + self.x_H2O(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"H2O") + self.x_CO2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"CO2") # First guess for h3
        nb_iter = 0
        while True:
            previous_h_3 = self.h_3
            self.lamda = (self.h_3 - self.x_N2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"N2") - self.x_H2O(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"H2O") - self.x_CO2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"CO2"))/(self.x_O2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"O2"))
            self.h_3 = self.x_O2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"O2") + self.x_N2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"N2") + self.x_H2O(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"H2O") + self.x_CO2(self.lamda)*CP.PropsSI('H','P',self.p_3,'T',self.T_3,"CO2")
            nb_iter += 1
            if abs(self.h_3 - previous_h_3) < 1e-6 or nb_iter > 10:
                break
            # new gas composition
        self.gas = ['O2','N2','H2O','CO2']
        self.gas_prop = [self.x_O2(self.lamda),self.x_N2(self.lamda),self.x_H2O(self.lamda),self.x_CO2(self.lamda)]
        self.gas_string = 'SRK::'
        for i in range(len(self.gas)):
            self.gas_string += self.gas[i] + '[' + str(self.gas_prop[i]) + ']'
            if i < len(self.gas)-1:
                self.gas_string += '&'
        self.s_3 = CP.PropsSI('S','P',self.p_3,'T',self.T_3,self.gas_string) # [J/kg/K] entropy at state 3
        self.h_0_gas = self.x_H2O(self.lamda)*CP.PropsSI('H','P',self.p_0_H20,'T',self.T_0,"H2O") + self.x_CO2(self.lamda)*CP.PropsSI('H','P',self.p_0_air,'T',self.T_0,"CO2") + self.x_N2(self.lamda)*CP.PropsSI('H','P',self.p_0_air,'T',self.T_0,"N2") + self.x_O2(self.lamda)*CP.PropsSI('H','P',self.p_0_air,'T',self.T_0,"O2") # [J/kg] enthalpy at state 3
        self.s_0_gas = self.x_H2O(self.lamda)*CP.PropsSI('S','P',self.p_0_H20,'T',self.T_0,"H2O") + self.x_CO2(self.lamda)*CP.PropsSI('S','P',self.p_0_air,'T',self.T_0,"CO2") + self.x_N2(self.lamda)*CP.PropsSI('S','P',self.p_0_air,'T',self.T_0,"N2") + self.x_O2(self.lamda)*CP.PropsSI('S','P',self.p_0_air,'T',self.T_0,"O2") # [J/kg/K] entropy at state 3
        self.e_3 = (self.h_3-self.h_0_gas) - self.T_0*(self.s_3-self.s_0_gas) # [J/kg] exergy at state 3
        # State 4 -------------------------------------------------------------
        self.p_4 = self.p_1
        gamma_t_in = CP.PropsSI('CPMASS','P',self.p_3,'T',self.T_3,self.gas_string)/CP.PropsSI('CVMASS','P',self.p_3,'T',self.T_3,self.gas_string) 
        self.T_4 = self.T_3*(self.p_4/self.p_3)**(((gamma_t_in-1)*self.eta_pi_t)/(gamma_t_in))
        nb_iter = 0
        while True:
            previous_T_4 = self.T_4
            T_vals = np.linspace(self.T_3,self.T_4,self.integration_points)
            gamma_t_moy = simpson(CP.PropsSI('CPMASS','P',self.p_3,'T',T_vals,self.gas_string)/CP.PropsSI('CVMASS','P',self.p_3,'T',T_vals,self.gas_string), x=T_vals)/(self.T_4-self.T_3)
            self.T_4 = self.T_3*(self.p_4/self.p_3)**(((gamma_t_moy-1)*self.eta_pi_t)/(gamma_t_moy))
            nb_iter += 1
            if abs(self.T_4 - previous_T_4) < 1e-6 or nb_iter > 10:
                break
        self.h_4 = CP.PropsSI('H','P',self.p_4,'T',self.T_4,self.gas_string) # [J/kg] enthalpy at state 4
        self.s_4 = CP.PropsSI('S','P',self.p_4,'T',self.T_4,self.gas_string) # [J/kg/K] entropy at state 4
        self.e_4 = (self.h_4-self.h_0_gas) - self.T_0*(self.s_4-self.s_0_gas) # [J/kg] exergy at state 4


        # States --------------------------------------------------------------
        self.p           = self.p_1, self.p_2, self.p_3, self.p_4
        self.T           = self.T_1, self.T_2, self.T_3, self.T_4
        self.s           = self.s_1, self.s_2, self.s_3, self.s_4
        self.h           = self.h_1, self.h_2, self.h_3, self.h_4
        self.e           = self.e_1, self.e_2, self.e_3, self.e_4
        self.DAT         = self.p,self.T,self.s,self.h,self.e
        # Combustion paramters ------------------------------------------------
        self.COMBUSTION  = self.LHV,self.e_f,self.excess_air,self.gas,self.gas_prop
        # Mass flow rates -----------------------------------------------------
        self.MASSFLOW    = self.dotm_a,self.dotm_f,self.dotm_g
        # Efficiencies --------------------------------------------------------
        self.ETA         = self.eta_cyclen,self.eta_toten,self.eta_mec,self.eta_cyclex,self.eta_totex,self.eta_rotex,self.eta_combex
        # Energy losses -------------------------------------------------------
        self.DATEN       = self.loss_mec,self.loss_echen
        # Exergy losses -------------------------------------------------------
        self.DATEX       = self.loss_mec,self.loss_rotex,self.loss_combex,self.loss_echex
        # Cycle work ----------------------------------------------------------
        self.WORK        = self.W_mcy
        # Energy and Exergy pie charts
        if self.display: self.FIG = self.fig_pie_en,self.fig_pie_ex, self.fig_Ts, self.fig_ph