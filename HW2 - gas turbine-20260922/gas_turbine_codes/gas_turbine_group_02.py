"""
LELME2150 - Thermal cycles
Homework 2 - Gas turbine

Group 02

AI-assistance
------------------------
This code was written by the group. The following AI tools were used as support:
  - Claude (Anthropic), in a chat conversation: to discuss the equations,
    to draft some parts of the code and to help debug.
  - GitHub Copilot (VS Code): inline auto-completion suggestions while typing, docstring generation.
All AI-generated suggestions were reviewed, adapted and tested by the group. The results
were checked against book results (p. 125) of the assignment and the exergy pie chart of the course.
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

    # Fuel properties for different fuels obtained from the book "Thermal power plants: Energetic and exergetic approaches" by D.Johnson, J.Martin and P.Wauters
    FUELS = {'Methane': {
        'alkane': [1, 4],
        'LHV': 50150*1e3,       # J/kg
        'HHV': 55695*1e3,       # J/kg
        'e_f': 52215*1e3,       # J/kg
    },
    'Propane': {
        'alkane': [3, 8],
        'LHV': 46465*1e3,       # J/kg
        'HHV': 50495*1e3,       # J/kg
        'e_f': 49045*1e3,       # J/kg
    },
    'Hydrogen': {
        'alkane': [0, 2],
        'LHV': 120900*1e3,      # J/kg
        'HHV': 142540*1e3,      # J/kg
        'e_f': 118790*1e3,      # J/kg
    }}
    
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

        # Compute the molar mass of the fuel and the air
        self.R = CP.PropsSI('GAS_CONSTANT','O2') # gas constant
        M_m_O = CP.PropsSI('MOLARMASS','O2')/2 # [kg/mol] molar mass of Oxygen
        M_m_C = CP.PropsSI('MOLARMASS','CO2')-(2*M_m_O) # [kg/mol] molar mass of Carbon
        M_m_H = CP.PropsSI('MOLARMASS','H2')/2 # [kg/mol] molar mass of Hydrogen
        M_m_air = CP.PropsSI('MOLARMASS','N2')*self.x_N2_0 + CP.PropsSI('MOLARMASS','O2')*self.x_O2_0 # [kg/mol] molar mass of air
        M_m_f   = self.x*M_m_C + self.y*M_m_H # masse d'une mole de combustible (kg/mol)
        n_a1  = (self.x + self.y/4)/self.x_O2_0    # mol d'air / mol de combustible
        self.ma1 = n_a1*M_m_air / M_m_f         # kg air / kg combustible

        
        # Set the reference state for the enthalpy and entropy calculations
        self.T_0, self.p_0 = 288.15, 1e5
        for f in ['N2','O2','CO2']:
            CP.set_reference_state(f, self.T_0, CP.PropsSI('Dmolar','T',self.T_0,'P',self.p_0,f), 0, 0)
        CP.set_reference_state('Water', self.T_0, CP.PropsSI('Dmolar','T',self.T_0,'Q',1,'Water'), 0, 0)
        # LHV, HHV and e_f
        matches = [name for name, props in self.FUELS.items() if props['alkane'] == self.alkane]

        if len(matches) != 1:
            raise ValueError(
                f"Combustible non reconnu ou ambigu : {self.alkane}"
            )
        self.fuel = matches[0]
        self.LHV = gas_turbine.FUELS[self.fuel]['LHV'] # [J/kg] Lower heating value of the fuel
        self.HHV = gas_turbine.FUELS[self.fuel]['HHV'] # [J/kg] Higher heating value of the fuel
        self.e_f = gas_turbine.FUELS[self.fuel]['e_f'] # [J/kg] exergy of the fuel
        # h_f
        self.h_f = (CP.PropsSI('H','P',self.p_0,'T',self.T_1,self.fuel)  - CP.PropsSI('H','P',self.p_0,'T',self.T_0,self.fuel))   # specific enthalpy (we suppose that fuel is injected at T_1, ambiant temperature) [J/kg]
        # Set the number of points for the numerical integrations and the display option
        self.integration_points = 20
        self.display            = display                   # if True, make the plots

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
    def gamma_air(self, p, T):
        x = {'O2':self.x_O2_0,'N2':self.x_N2_0}
        M_mix = sum(x[k]*CP.PropsSI('MOLARMASS',k) for k in x)
        Cp = sum(x[k]*CP.PropsSI('CPMOLAR','P',p,'T',T,k) for k in x)/M_mix
        Cv = sum(x[k]*CP.PropsSI('CVMOLAR','P',p,'T',T,k) for k in x)/M_mix
        return Cp/Cv
    def gamma_gas(self, p, T, lam):
        """
        Compute the specific heat ratio of the combustion products
        """
        x = {'O2':self.x_O2(lam),'N2':self.x_N2(lam),'H2O':self.x_H2O(lam),'CO2':self.x_CO2(lam)}
        M_mix = sum(x[k]*CP.PropsSI('MOLARMASS',k) for k in x)
        Cp = sum(x[k]*CP.PropsSI('CPMOLAR','P',p,'T',T,k) for k in x)/M_mix
        Cv = sum(x[k]*CP.PropsSI('CVMOLAR','P',p,'T',T,k) for k in x)/M_mix
        return Cp/Cv
    def h_air(self, p, T):
        """
        Compute the specific enthalpy of the air
        """
        x = {'O2':self.x_O2_0,'N2':self.x_N2_0}
        M_mix = sum(x[k]*CP.PropsSI('MOLARMASS',k) for k in x)
        return sum(x[k]*CP.PropsSI('Hmolar','P',p,'T',T,k) for k in x)/M_mix   # [J/kg]
    def s_air(self, p, T):
        """
        Compute the specific entropy of the air
        """
        x = {'O2':self.x_O2_0,'N2':self.x_N2_0}
        M_mix = sum(x[k]*CP.PropsSI('MOLARMASS',k) for k in x)
        return sum(x[k]*CP.PropsSI('Smolar','P',p,'T',T,k) for k in x)/M_mix   # [J/kg/K]
    def Hm(self, k, p, T):
        """[J/mol] enthalpie molaire, l'eau est forcée en vapeur sous sa T de saturation"""
        if k == 'H2O' and T < CP.PropsSI('T','P',p,'Q',1,'Water'):
            return CP.PropsSI('Hmolar','P',1.0,'T',T,'Water')      
        return CP.PropsSI('Hmolar','P',p,'T',T,k)
    def Sm(self, k, p, T):
        """[J/mol/K] entropie molaire, même convention"""
        if k == 'H2O' and T < CP.PropsSI('T','P',p,'Q',1,'Water'):
            return CP.PropsSI('Smolar','P',1.0,'T',T,'Water') - self.R*np.log(p/1.0)
        return CP.PropsSI('Smolar','P',p,'T',T,k)
    def h_gas(self, p, T, lam):
        """
        Compute the specific enthalpy of the combustion products
        """
        x = {'O2':self.x_O2(lam),'N2':self.x_N2(lam),'H2O':self.x_H2O(lam),'CO2':self.x_CO2(lam)}
        M_mix = sum(x[k]*CP.PropsSI('MOLARMASS',k) for k in x)
        return sum(x[k]*self.Hm(k, p, T) for k in x)/M_mix   # [J/kg]
    def s_gas(self, p, T, lam):
        """
        Compute the specific entropy of the combustion products
        """
        x = {'O2':self.x_O2(lam),'N2':self.x_N2(lam),'H2O':self.x_H2O(lam),'CO2':self.x_CO2(lam)}
        M_mix = sum(x[k]*CP.PropsSI('MOLARMASS',k) for k in x)
        return sum(x[k]*self.Sm(k, p, T) for k in x)/M_mix   # [J/kg/K] 
    def _poly_path(self, p_a, T_a, p_b, T_b, N):
        """Chemin polytropique T = T_a (p/p_a)^n passant par (p_a,T_a) et (p_b,T_b)"""
        n = np.log(T_b/T_a)/np.log(p_b/p_a)
        p = np.geomspace(p_a, p_b, N)
        return p, T_a*(p/p_a)**n
    def make_figures(self, N=60):
        lam = self.excess_air
        ev  = lambda f, p, T: np.array([f(a, b) for a, b in zip(p, T)])
        s_a = lambda p, T: self.s_air(p, T)
        h_a = lambda p, T: self.h_air(p, T)
        s_g = lambda p, T: self.s_gas(p, T, lam)
        h_g = lambda p, T: self.h_gas(p, T, lam)
        w = np.linspace(0, 1, N)

        # 1-2 compression (air)
        p12, T12 = self._poly_path(self.p_1, self.T_1, self.p_2, self.T_2, N)
        s12, h12 = ev(s_a, p12, T12), ev(h_a, p12, T12)
        # 2-3 combustion (air -> fumées)
        p23 = self.p_2 + w*(self.p_3 - self.p_2)
        T23 = self.T_2 + w*(self.T_3 - self.T_2)
        s23 = (1-w)*ev(s_a, p23, T23) + w*ev(s_g, p23, T23)
        h23 = (1-w)*ev(h_a, p23, T23) + w*ev(h_g, p23, T23)
        # 3-4 détente (fumées)
        p34, T34 = self._poly_path(self.p_3, self.T_3, self.p_4, self.T_4, N)
        s34, h34 = ev(s_g, p34, T34), ev(h_g, p34, T34)
        # 4-1 échappement fictif (fumées -> air), isobare
        p41 = np.full(N, self.p_1)
        T41 = self.T_4 + w*(self.T_1 - self.T_4)
        s41 = (1-w)*ev(s_g, p41, T41) + w*ev(s_a, p41, T41)
        h41 = (1-w)*ev(h_g, p41, T41) + w*ev(h_a, p41, T41)

        # --- T-s ---
        fig_Ts, ax = plt.subplots()
        for s, T, ls in [(s12,T12,'-'), (s23,T23,'-'), (s34,T34,'-'), (s41,T41,'--')]:
            ax.plot(s/1e3, T, ls, color='tab:blue')
        ax.plot(np.array(self.s)/1e3, self.T, 'o', color='tab:red')
        for i in range(4):
            ax.annotate(str(i+1), (self.s[i]/1e3, self.T[i]), textcoords='offset points', xytext=(6,6))
        ax.set_xlabel('s [kJ/kg/K]'); ax.set_ylabel('T [K]'); ax.set_title('T-s diagram'); ax.grid(True)

        # --- p-h ---
        fig_ph, ax = plt.subplots()
        for h, p, ls in [(h12,p12,'-'), (h23,p23,'-'), (h34,p34,'-'), (h41,p41,'--')]:
            ax.plot(h/1e3, p/1e5, ls, color='tab:blue')
        ax.plot(np.array(self.h)/1e3, np.array(self.p)/1e5, 'o', color='tab:red')
        for i in range(4):
            ax.annotate(str(i+1), (self.h[i]/1e3, self.p[i]/1e5), textcoords='offset points', xytext=(6,6))
        ax.set_yscale('log')
        ax.set_xlabel('h [kJ/kg]'); ax.set_ylabel('p [bar]'); ax.set_title('p-h diagram'); ax.grid(True, which='both')

        # --- camembert énergie ---
        P_prim_en = self.dotm_f*(self.LHV + self.h_f)     # [W] flux d'énergie primaire
        loss_en = [self.P_e, self.loss_mec, self.loss_echen]
        names_en = ['Net power', 'Mechanical losses', 'Exhaust losses']
        fig_pie_en, ax = plt.subplots()
        ax.pie(loss_en, labels=[f'{n}\n{v/1e6:.1f} MW' for n, v in zip(names_en, loss_en)], autopct='%1.1f%%')
        ax.set_title(f'Primary energy flow {P_prim_en/1e6:.1f} MW')

        # --- camembert exergie ---
        P_prim_ex = self.dotm_f*self.e_f                  # [W] flux d'exergie primaire
        loss_ex = [self.P_e, self.loss_mec, self.loss_rotex, self.loss_combex, self.loss_echex]
        names_ex = ['Net power', 'Mechanical losses', 'Rotor losses', 'Combustion losses', 'Exhaust losses']
        fig_pie_ex, ax = plt.subplots()
        ax.pie(loss_ex, labels=[f'{n}\n{v/1e6:.1f} MW' for n, v in zip(names_ex, loss_ex)], autopct='%1.1f%%')
        ax.set_title(f'Primary exergy flow {P_prim_ex/1e6:.1f} MW')

        return fig_pie_en, fig_pie_ex, fig_Ts, fig_ph
    def evaluate(self):
        """
        This is the main method of the gas turbine class.
        It evaluates the different thermodynamic quantities at each state of 
        the cycle, as well as some KPI's.
        """
        # State 1 -------------------------------------------------------------
        self.h_1 = self.h_air(self.p_1, self.T_1)  # [J/kg] enthalpy at state 1
        self.s_1 = self.s_air(self.p_1, self.T_1)  # [J/kg/K] entropy at state 1
        self.e_1 = (self.h_1) - self.T_0*(self.s_1) # [J/kg] exergy at state 1
        # State 2 -------------------------------------------------------------
        self.p_2 = self.r_c*self.p_1
        gamma_c_in = self.gamma_air(self.p_1, self.T_1) # [-] specific heat ratio of the air at state 1 (used for the first guess of T2)
            # Fixed point iteration to find T2
        self.T_2 = self.T_1*(self.p_2/self.p_1)**((gamma_c_in-1)/(gamma_c_in*self.eta_pi_c)) # First guess for T2
        nb_iter = 0
        while True:
            previous_T_2 = self.T_2
            T_vals = np.linspace(self.T_1,self.T_2,self.integration_points)
            gamma_c_moy = simpson(self.gamma_air(self.p_1, T_vals), x=T_vals)/(self.T_2-self.T_1)
            self.T_2 = self.T_1*(self.p_2/self.p_1)**((gamma_c_moy-1)/(gamma_c_moy*self.eta_pi_c))
            nb_iter += 1
            if abs(self.T_2 - previous_T_2) < 1e-6 or nb_iter > 10:
                break
            # ------------------------------------
        self.h_2 = self.h_air(self.p_2, self.T_2) # [J/kg] enthalpy at state 2
        self.s_2 = self.s_air(self.p_2, self.T_2) # [J/(kg.K)] entropy at state 2
        self.e_2 = (self.h_2) - self.T_0*(self.s_2) # [J/kg] exergy at state 2
        # State 3 -------------------------------------------------------------
        self.p_3 = self.p_2*self.k_cc
            # Fixed point iteration to find h3
        self.excess_air = 1 # First guess for excess air
        self.h_3 = self.h_gas(self.p_3, self.T_3, self.excess_air) # First guess for h3
        nb_iter = 0
        while True:
            previous_h_3 = self.h_3
            self.excess_air = (self.LHV+self.h_f-self.h_3)/(self.ma1*(self.h_3-self.h_2)) # [-] excess air ratio
            self.h_3 = self.h_gas(self.p_3, self.T_3, self.excess_air) # [J/kg] enthalpy at state 3
            nb_iter += 1
            if abs(self.h_3 - previous_h_3) < 1e-6 or nb_iter > 100:
                break
        self.gas      = ['O2','N2','H2O','CO2']
        self.gas_prop = [self.x_O2(self.excess_air), self.x_N2(self.excess_air),
                         self.x_H2O(self.excess_air), self.x_CO2(self.excess_air)]
        self.s_3 = self.s_gas(self.p_3, self.T_3, self.excess_air) # [J/kg/K] entropy at state 3
        s_0_gas = self.s_gas(self.p_0, self.T_0, self.excess_air) # [J/kg/K] entropy of the gas at reference conditions
        h_0_gas = self.h_gas(self.p_0, self.T_0, self.excess_air) # [J/kg] enthalpy of the gas at reference conditions
        self.e_3 = (self.h_3 - h_0_gas) - self.T_0*(self.s_3 - s_0_gas) # [J/kg] exergy at state 3
        # State 4 -------------------------------------------------------------
        self.p_4 = self.p_1
        gamma_t_in = self.gamma_gas(self.p_3, self.T_3, self.excess_air) # [-] specific heat ratio of the combustion products at state 3 (used for the first guess of T4)
        self.T_4 = self.T_3*(self.p_4/self.p_3)**(((gamma_t_in-1)*self.eta_pi_t)/(gamma_t_in))
        nb_iter = 0
        while True:
            previous_T_4 = self.T_4
            T_vals = np.linspace(self.T_3,self.T_4,self.integration_points)
            gamma_t_moy = simpson(self.gamma_gas(self.p_3, T_vals, self.excess_air), x=T_vals)/(self.T_4-self.T_3)
            self.T_4 = self.T_3*(self.p_4/self.p_3)**(((gamma_t_moy-1)*self.eta_pi_t)/(gamma_t_moy))
            nb_iter += 1
            if abs(self.T_4 - previous_T_4) < 1e-6 or nb_iter > 10:
                break
        self.h_4 = self.h_gas(self.p_4, self.T_4, self.excess_air)    # [J/kg] enthalpy at state 4
        self.s_4 = self.s_gas(self.p_4, self.T_4, self.excess_air)    # [J/kg/K] entropy at state 4
        self.e_4 = (self.h_4 - h_0_gas) - self.T_0*(self.s_4 - s_0_gas)    # [J/kg] exergy at state 4

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
        f_ratio = 1/(self.excess_air*self.ma1)                  # [-] mdot_f / mdot_a
        w_c     = self.h_2 - self.h_1                           # [J/kg air] compressor work
        w_t     = (1+f_ratio)*(self.h_3 - self.h_4)             # [J/kg air] turbine work
        self.W_mcy    = w_t - w_c                               # [J/kg air] cycle work
    
        self.dotm_a = self.P_e/(w_t*(1-self.k_mec)-w_c*(1+self.k_mec))    # [kg/s] air 
        self.dotm_f = f_ratio*self.dotm_a                       # [kg/s] fuel
        self.dotm_g = self.dotm_a + self.dotm_f                 # [kg/s] gas
        self.MASSFLOW    = self.dotm_a,self.dotm_f,self.dotm_g
        # Efficiencies --------------------------------------------------------
        P_mcy = self.dotm_g*(self.h_3-self.h_4)-self.dotm_a*(self.h_2-self.h_1) # [W] cycle work
            # Energy efficiencies ---------------------------------------------------
        self.eta_cyclen  = P_mcy/(self.dotm_g*self.h_3 - (self.dotm_a*self.h_2))    # [-] cycle energy efficiency (Course 1 slides 6)
        self.eta_mec   = self.P_e/P_mcy                             # [-] mechanical efficiency (p.122)
        self.eta_toten   = self.eta_cyclen*self.eta_mec                   # [-] total energy efficiency
            # Exergy efficiencies ---------------------------------------------------
        self.eta_cyclex  = P_mcy/(self.dotm_g*self.e_3 - self.dotm_a*self.e_2)    # [-] cycle exergy efficiency (p.124)
        self.eta_totex   = self.P_e/(self.dotm_f*self.e_f)                          # [-] total exergy efficiency (p.124)
        self.eta_rotex   = P_mcy/(self.dotm_g*(self.e_3-self.e_4) - self.dotm_a*(self.e_2-self.e_1)) # [-] rotor exergy efficiency (p.124)
        self.eta_combex  = (self.dotm_g*self.e_3 - self.dotm_a*self.e_2)/(self.dotm_f*self.e_f) # [-] combustion exergy efficiency (p.124)
        self.ETA         = self.eta_cyclen,self.eta_toten,self.eta_mec,self.eta_cyclex,self.eta_totex,self.eta_rotex,self.eta_combex
        # Energy losses -------------------------------------------------------
        self.loss_mec    = P_mcy - self.P_e # [W] mechanical losses
        self.loss_echen  = self.dotm_g*self.h_4 - self.dotm_a*self.h_1 # [W] energy losses in the exhaust gases
        self.DATEN       = self.loss_mec,self.loss_echen
        # Exergy losses -------------------------------------------------------
        exergy_rotor     = self.dotm_g*(self.e_3-self.e_4) - self.dotm_a*(self.e_2-self.e_1) # [W] total exergy losses in the rotor
        self.loss_rotex  = exergy_rotor - P_mcy # [W] real exergy losses in the rotor (total losses - work produced)
        self.loss_combex = self.dotm_a*self.e_2 + self.dotm_f*self.e_f - self.dotm_g*self.e_3 # [W] exergy losses in the combustion chamber
        self.loss_echex  = self.dotm_g*self.e_4 - self.dotm_a*self.e_1 # [W] exergy losses in the exhaust gases
        self.DATEX       = self.loss_mec,self.loss_rotex,self.loss_combex,self.loss_echex
        # Cycle work ----------------------------------------------------------
        self.WORK        = self.W_mcy
        # Energy and Exergy pie charts
        if self.display: 
            self.fig_pie_en, self.fig_pie_ex, self.fig_Ts, self.fig_ph = self.make_figures()
            self.FIG = self.fig_pie_en,self.fig_pie_ex, self.fig_Ts, self.fig_ph
        else:
            self.FIG = None, None, None, None
