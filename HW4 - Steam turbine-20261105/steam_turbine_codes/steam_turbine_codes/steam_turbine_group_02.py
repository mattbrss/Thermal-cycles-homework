"""
LELME2150 - Thermal Cycles
Homework 4 - Steam Turbine      

Signature of the function for the steam turbine

"""

#
# === IMPORT PACKAGES ==========================================================
#

import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt

#
# === STEAM TURBINE ============================================================
#

class steam_turbine(object):
    """
    Class for the simulation of the three-body steam turbine cycle with
    one reheater, one deaerator, and three feed-water extractions.

    Parameters
    ----------
    inputs : float
        Electrical power output P_e [W].
    parameters : dict
        Dictionary of model parameters (SI units: K and Pa).
    display : bool
        If True, generate T-s / h-s diagrams and energy/exergy pie charts.

    Methods
    -------
    evaluate()
        Compute all thermodynamic states and performance indicators.
    """

    def __init__(self, inputs, parameters, display):
        """Create a steam turbine object."""
        P_e                  = inputs
        self.P_e             = P_e                               # [W]  electrical power output

        self.T_ref           = parameters['T_ref']               # [K]  reference temperature for exergy
        self.p_ref           = parameters['p_ref']               # [Pa] reference pressure for exergy

        self.p_6             = parameters['p_6']                 # [Pa] pressure at state 6
        self.p_8             = parameters['p_8']                 # [Pa] maximum steam pressure (state 8)
        self.p_9             = parameters['p_9']                 # [Pa] reheating pressure (state 9)

        self.T_drum          = parameters['T_drum']              # [K]  temperature inside the drum
        self.T_max           = parameters['T_max']               # [K]  maximum steam temperature

        self.T_cd_out        = parameters['T_cd_out']            # [K]  condenser cold outlet temperature of the secondary fluid
        self.T_pinch_cd      = parameters['T_pinch_cd']          # [K]  pinch temperature at the condenser
        self.T_pinch_ex      = parameters['T_pinch_ex']          # [K]  pinch temperature at the heat exchangers

        self.x_11min         = parameters['x_min']               # [-]  min vapour quality at LP exit

        self.eta_is_HP       = parameters['eta_is_HP']           # [-]  HP turbine isentropic efficiency
        self.eta_is_LP       = parameters['eta_is_LP']           # [-]  LP turbine isentropic efficiency
        self.eta_mec         = parameters['eta_mec']             # [-]  mechanical efficiency (all shafts)
        self.eta_pump        = parameters['eta_pump']            # [-]  pump internal efficiency
        self.k_heat          = parameters['k_heat']              # [-]  pressure loss at steam generator/reheater

        self.display         = display                           # If True, display the plots

    def evaluate(self):
        """
        This is the main method of the steam turbine class.
        It evaluates the different thermodynamic quantities at each state of 
        the cycle, as well as some KPI's based on several 
        inputs and for a given electrical production. It can also plot the T-s 
        and h-s as well as the energy and exergy pies.
        """

        # initial values: p6,p8,p9
        # REFERENCE STATE
        h_ref = CP.PropsSI('H','T',self.T_ref,'P',self.p_ref,'Water')
        s_ref = CP.PropsSI('S','T',self.T_ref,'P',self.p_ref,'Water')
        #build a state
        def State(p,T=None,x=None,h=None):
            if x is not None:
                h = CP.PropsSI('H','P',p,'Q',x,"Water")
            elif T is not None:
                h = CP.PropsSI('H','P',p,'T',T,"Water")
            T_result = CP.PropsSI('T','P',p,'H',h,"Water")
            s_result = CP.PropsSI('S','P',p,'H',h,"Water")
            if p < CP.PropsSI('Pcrit','Water'):
                hl=CP.PropsSI('H','P',p,'Q',0,"Water")
                hv=CP.PropsSI('H','P',p,'Q',1,"Water")
                hlv=hv-hl
                x_result = (h-hl)/hlv
                if x_result < -1e-9 or x_result > 1+1e-9:
                    x_result=-1
            else:
                x_result = -1
            e_result = (h-h_ref)-self.T_ref*(s_result-s_ref)
            return p,T_result,h,s_result,x_result,e_result

        # STATE 1
        # as we have here liquid (saturated) water, then x1=0

        self.T_1 = self.T_cd_out+self.T_pinch_cd
        self.p_1 = CP.PropsSI('P','T',self.T_1,'Q',0,"Water")
        self.x_1 = 0
        _,_,self.h_1,self.s_1,_,self.e_1 = State(self.p_1,x=self.x_1)
        rho_1 = CP.PropsSI('D','P',self.p_1,'H',self.h_1,"Water")
        v_1 = 1/rho_1

        # STATE 2
        # we have a pump between states 1 and 2, so we have to use its efficiency, known as eta_pump, defined as :
        # eta_pump = w_u/w_m = (v Delta p)/(Delta h) where we can obtain h2
        # to complete the state 2, we have to know that pdrum = p4 = psat(Tdrum) = p3 = p2

        self.p_4 = CP.PropsSI('P','T',self.T_drum,'Q',0,"Water")
        self.p_2 = self.p_4
        self.h_2 = self.h_1+v_1*((self.p_2-self.p_1)/self.eta_pump)
        _,self.T_2,_,self.s_2,self.x_2,self.e_2 = State(self.p_2,h=self.h_2)

        # STATE 4
        # we already know p4 and, as it is a saturated liquid, we also have x4=0

        _,self.T_4,self.h_4,self.s_4,self.x_4,self.e_4 = State(self.p_4,x=0)
        rho_4 = CP.PropsSI('D','P',self.p_4,'H',self.h_4,"Water")
        v_4 = 1/rho_4

        # STATE 5
        # as there is a heat exchanger between 5 and 6, we can say, by using an isobaric assumption, that p5 = p6
        # (bc p6 is given)

        self.p_5 = self.p_6
        self.h_5 = self.h_4+v_4*((self.p_5-self.p_4)/self.eta_pump)
        _,self.T_5,_,self.s_5,self.x_5,self.e_5 = State(self.p_5,h=self.h_5)

        # STATE 8
        # we know p8, and by seeing the system we can tell that T8 = Tmax
        # (after that state, we have expansion etc. such that pressure and temperature decrease)

        self.T_8 = self.T_max
        _,_,self.h_8,self.s_8,self.x_8,self.e_8 = State(self.p_8,T=self.T_8)

        # STATE 9
        # we can use the turbine isentropic efficiency to obtain h9
        # then we can obtain the rest of the state

        s_9s = self.s_8
        h_9s = CP.PropsSI('H','P',self.p_9,'S',s_9s,"Water")
        self.h_9 = self.h_8 - self.eta_is_HP*(self.h_8-h_9s)
        _,self.T_9,_,self.s_9,self.x_9,self.e_9 = State(self.p_9,h=self.h_9)

        # STATE 10
        # the description gives us that p10 = p9*kheat
        # like 8, 10 is also at maximum temperature, before going to the LP turbine

        self.p_10 = self.p_9*self.k_heat
        self.T_10 = self.T_max
        _,_,self.h_10,self.s_10,self.x_10,self.e_10 = State(self.p_10,T=self.T_10)

        # STATE 12
        # as the heat exchanger is isobaric, we have p12 = p9
        # this state is with saturated liquid, so x12 = 0

        self.p_12 = self.p_9
        self.x_12 = 0
        _,self.T_12,self.h_12,self.s_12,_,self.e_12 = State(self.p_12,x=self.x_12)

        # STATE 13
        # we can directly say that h13 = h12 because V1 is isenthalpic
        # and it works at the pressure of the drum, so p13 = p4

        self.p_13 = self.p_4
        self.h_13 = self.h_12
        _,self.T_13,_,self.s_13,self.x_13,self.e_13 = State(self.p_13,h=self.h_13)

        # TO VERIFY : Hypothesis by myself
        # we just know that p11 < p16 < p14 < p10
        # by using some ratios :
        # (h14-h11)/(h10-h11) = 2/3
        # (h16-h11)/(h10-h11) = 1/3
        
        self.p_11 = self.p_1
        h_11s = CP.PropsSI('H','P',self.p_11,'S',self.s_10,"Water")
        self.h_11  = self.h_10-self.eta_is_LP*(self.h_10-h_11s)
        self.h_14  = self.h_11+(2/3)*(self.h_10-self.h_11)
        self.h_16  = self.h_11+(1/3)*(self.h_10-self.h_11)

        def p_from_h(h_target):
            """Pressure at which the real LP expansion line (eta_is_LP from state 10) reaches h_target."""
            lo,hi = self.p_1,self.p_10
            for _ in range(60):                       # bisection (h increases with p on the line)
                mid   = np.sqrt(lo*hi)
                h_mid = self.h_10-self.eta_is_LP*(self.h_10-CP.PropsSI('H','P',mid,'S',self.s_10,"Water"))
                if h_mid < h_target:
                    lo = mid
                else:
                    hi = mid
            return np.sqrt(lo*hi)

        self.p_14 = p_from_h(self.h_14)
        self.p_16 = p_from_h(self.h_16)

        _,self.T_14,self.h_14,self.s_14,self.x_14,self.e_14 = State(self.p_14,h=self.h_14)
        _,self.T_16,self.h_16,self.s_16,self.x_16,self.e_16 = State(self.p_16,h=self.h_16)
        _,self.T_11,self.h_11,self.s_11,self.x_11,self.e_11 = State(self.p_11,h=self.h_11)

        # STATE 15
        # as V2 is isenthalpic, we have that h14 = h15 and we can also take p15 = pdrum

        self.p_15 = self.p_4
        self.h_15 = self.h_14
        _,self.T_15,_,self.s_15,self.x_15,self.e_15 = State(self.p_15,h=self.h_15)

        # STATE 11 (CORRECTION BY THE LP TURBINE)

        # s_11s = self.s_16
        # h_11s = CP.PropsSI('H','P',self.p_11,'S',s_11s,"Water")
        # self.h_11 = self.h_16-self.eta_is_LP*(self.h_16-h_11s)
        # _,self.T_11,_,self.s_11,self.x_11,self.e_11 = State(self.p_11,h=self.h_11)

        # STATE 17
        # as we know that this state is with saturated liquired, so x17 = 0
        # and by assuming that the heat exchanger is isobaric

        self.p_17 = self.p_16
        self.x_17 = 0
        _,self.T_17,self.h_17,self.s_17,_,self.e_17 = State(self.p_17,x=self.x_17)

        # STATE 18
        # we can calculate this state by using the fact that there is V3 between 17 and 18
        # so we have that h18 = h17
        # and we can admit that p18 = p1

        self.p_18 = self.p_1
        self.h_18 = self.h_17
        _,self.T_18,_,self.s_18,self.x_18,self.e_18 = State(self.p_18,h=self.h_18)

        # PARTICULAR STATES 3,6,7
        
        # heat exchanger R1 (STATE 3)

        Tsat_16 = CP.PropsSI('T','P',self.p_16,'Q',0,"Water")
        hg_16 = CP.PropsSI('H','P',self.p_16,'Q',1,"Water")
        h_start = min(self.h_16,hg_16)
        h_d = CP.PropsSI('H','P',self.p_2,'T',Tsat_16-self.T_pinch_ex,"Water")
        self.X_3 = (h_d-self.h_2)/(h_start-self.h_17)
        self.p_3 = self.p_2
        self.h_3 = self.h_2+self.X_3*(self.h_16-self.h_17)
        _,self.T_3,_,self.s_3,self.x_3,self.e_3 = State(self.p_3,h=self.h_3)

        # heat exchanger R2 (STATE 6 AND 7)

        Tsat_9 = CP.PropsSI('T','P',self.p_9,'Q',0,"Water")
        hg_9 = CP.PropsSI('H','P',self.p_9,'Q',1,"Water")
        h_start_2 = min(self.h_9,hg_9)
        h_d_2 = CP.PropsSI('H','P',self.p_6,'T',Tsat_9-self.T_pinch_ex,"Water")
        diff_h = h_d_2 - self.h_5
        A = np.array([[h_start_2-self.h_12-diff_h, -diff_h],
                        [self.h_13-self.h_4, self.h_15-self.h_4]])
        b = np.array([diff_h, self.h_4 - self.h_3])
        self.X_1,self.X_2 = np.linalg.solve(A,b)
        m_tot = 1+self.X_1+self.X_2

        self.h_6 = self.h_5 + self.X_1*(self.h_9-self.h_12)/m_tot
        _,self.T_6,_,self.s_6,self.x_6,self.e_6 = State(self.p_6,h=self.h_6)
        self.p_7 = self.p_8/self.k_heat
        rho_6 = CP.PropsSI('D','P',self.p_6,'H',self.h_6,"Water")
        v_6 = 1/rho_6
        self.h_7 = self.h_6+v_6*((self.p_7-self.p_6)/self.eta_pump)
        _,self.T_7,_,self.s_7,self.x_7,self.e_7 = State(self.p_7,h=self.h_7)


        print("Flows",self.X_1,self.X_2,self.X_3)
        print("State1",self.p_1,self.T_1,self.h_1,self.s_1,self.x_1,self.e_1)
        print("State2",self.p_2,self.T_2,self.h_2,self.s_2,self.x_2,self.e_2)
        print("State3",self.p_3,self.T_3,self.h_3,self.s_3,self.x_3,self.e_3)
        print("State4",self.p_4,self.T_4,self.h_4,self.s_4,self.x_4,self.e_4)
        print("State5",self.p_5,self.T_5,self.h_5,self.s_5,self.x_5,self.e_5)
        print("State6",self.p_6,self.T_6,self.h_6,self.s_6,self.x_6,self.e_6)
        print("State7",self.p_7,self.T_7,self.h_7,self.s_7,self.x_7,self.e_7)
        print("State8",self.p_8,self.T_8,self.h_8,self.s_8,self.x_8,self.e_8)
        print("State9",self.p_9,self.T_9,self.h_9,self.s_9,self.x_9,self.e_9)
        print("State10",self.p_10,self.T_10,self.h_10,self.s_10,self.x_10,self.e_10)
        print("State11",self.p_11,self.T_11,self.h_11,self.s_11,self.x_11,self.e_11)
        print("State12",self.p_12,self.T_12,self.h_12,self.s_12,self.x_12,self.e_12)
        print("State13",self.p_13,self.T_13,self.h_13,self.s_13,self.x_13,self.e_13)
        print("State14",self.p_14,self.T_14,self.h_14,self.s_14,self.x_14,self.e_14)
        print("State15",self.p_15,self.T_15,self.h_15,self.s_15,self.x_15,self.e_15)
        print("State16",self.p_16,self.T_16,self.h_16,self.s_16,self.x_16,self.e_16)
        print("State17",self.p_17,self.T_17,self.h_17,self.s_17,self.x_17,self.e_17)
        print("State18",self.p_18,self.T_18,self.h_18,self.s_18,self.x_18,self.e_18)
        



        ### OUTPUTS ###########################################################

        # States ---------------------------------------------------------------
        self.p  = (self.p_1,  self.p_2,  self.p_3,  self.p_4,self.p_5,  self.p_6,  self.p_7,  self.p_8,self.p_9,  self.p_10, self.p_11, self.p_12,self.p_13,self.p_14,self.p_15,self.p_16,self.p_17,self.p_18)   # [Pa]
        self.T  = (self.T_1,  self.T_2,  self.T_3,  self.T_4,self.T_5,  self.T_6,  self.T_7,  self.T_8,self.T_9,  self.T_10, self.T_11, self.T_12,self.T_13,self.T_14,self.T_15,self.T_16,self.T_17,self.T_18)   # [K]
        self.h  = (self.h_1,  self.h_2,  self.h_3,  self.h_4,self.h_5,  self.h_6,  self.h_7,  self.h_8,self.h_9,  self.h_10, self.h_11, self.h_12,self.h_13,self.h_14,self.h_15,self.h_16,self.h_17,self.h_18)   # [J/kg]
        self.s  = (self.s_1,  self.s_2,  self.s_3,  self.s_4,self.s_5,  self.s_6,  self.s_7,  self.s_8,self.s_9,  self.s_10, self.s_11, self.s_12,self.s_13,self.s_14,self.s_15,self.s_16,self.s_17,self.s_18)   # [J/kg/K]
        self.x  = (self.x_1,  self.x_2,  self.x_3,  self.x_4,self.x_5,  self.x_6,  self.x_7,  self.x_8,self.x_9,  self.x_10, self.x_11, self.x_12,self.x_13,self.x_14,self.x_15,self.x_16,self.x_17,self.x_18)   # [-]
        self.e  = (self.e_1,  self.e_2,  self.e_3,  self.e_4,self.e_5,  self.e_6,  self.e_7,  self.e_8,self.e_9,  self.e_10, self.e_11, self.e_12,self.e_13,self.e_14,self.e_15,self.e_16,self.e_17,self.e_18)   # [J/kg]
        self.DAT = self.p, self.T, self.h, self.s, self.x, self.e
        

        # # Mass flow rates ------------------------------------------------------
        # self.MASSFLOW  = self.m_dot_tot, self.m_dot_cond
        # #   m_dot_tot   [kg/s]  total mass flow (through boiler)
        # #   m_dot_cond  [kg/s]  mass flow at the inlet of the condenser

        # # Extraction fractions -------------------------------------------------
        # self.XMASSFLOW = self.X_1, self.X_2, self.X_3
        # #   X_1  [-]  fraction of condenser mass flow extracted at HP turbine exit to R2 
        # #   X_2  [-]  fraction of condenser massflow extracted at LP turbine to the drum 
        # #   X_3  [-]  fraction of condenser massflow extracted inside LP turbine to R1

        # # Efficiencies ---------------------------------------------------------
        # self.ETA = self.eta_cyclen, self.eta_cyclex, self.eta_rotex
        # #   eta_cyclen  [-]  cycle energy efficiency
        # #   eta_cyclex  [-]  cycle exergy efficiency
        # #   eta_rotex   [-]  turbines and pumps exergy efficiency

        # # Energy losses --------------------------------------------------------
        # self.DATEN = self.loss_mec, self.loss_conden
        # #   loss_mec    [W]  mechanical losses
        # #   loss_conden [W]  heat rejected to the cold source

        # # Exergy losses --------------------------------------------------------
        # self.DATEX = self.loss_mec, self.loss_rotex, self.loss_condex, self.loss_valvex
        # #   loss_mec    [W]  mechanical losses
        # #   loss_rotex  [W]  exergy destroyed in turbines and pumps
        # #   loss_condex [W]  exergy rejected/destroyed in condenser
        # #   loss_valvex [W]  exergy destroyed in valves

        # ── Figures -----------------------------------------------------------
        if self.display:

            # --- saturation dome (closed at the critical point)
            T_c   = CP.PropsSI('Tcrit','Water')
            rho_c = CP.PropsSI('rhomass_critical','Water')
            Td = np.linspace(274.0,T_c-0.5,200)
            s_liq = np.array([CP.PropsSI('S','T',t,'Q',0,'Water') for t in Td])/1e3
            s_vap = np.array([CP.PropsSI('S','T',t,'Q',1,'Water') for t in Td])/1e3
            h_liq = np.array([CP.PropsSI('H','T',t,'Q',0,'Water') for t in Td])/1e3
            h_vap = np.array([CP.PropsSI('H','T',t,'Q',1,'Water') for t in Td])/1e3
            s_c = CP.PropsSI('S','T',T_c,'D',rho_c,'Water')/1e3
            h_c = CP.PropsSI('H','T',T_c,'D',rho_c,'Water')/1e3
            Td    = np.append(Td,T_c)
            s_liq = np.append(s_liq,s_c)
            s_vap = np.append(s_vap,s_c)
            h_liq = np.append(h_liq,h_c)
            h_vap = np.append(h_vap,h_c)

            # --- segments drawn as curves (pressure linear between both ends)
            segs_curve = [(2,3),(5,6),(7,8),(9,10),(11,1),(9,12),(16,17),(15,4),(13,4),(3,4)]
            # --- segments drawn as straight lines (expansions, pumps, valves, mixing)
            segs_line  = [(1,2),(4,5),(6,7),(8,9),(10,14),(14,16),(16,11),
                          (12,13),(14,15),(17,18),(18,1)]
            # --- extraction / drain branches in dashed lines
            dashed = {(9,12),(12,13),(14,15),(16,17),(17,18),(13,4),(15,4),(18,1),(3,4)}

            def draw(ax,kind):
                for a,b_ in segs_curve:
                    frac = np.linspace(0,1,60)
                    pp = self.p[a-1]+frac*(self.p[b_-1]-self.p[a-1])
                    hh = self.h[a-1]+frac*(self.h[b_-1]-self.h[a-1])
                    ss = [CP.PropsSI('S','P',pi,'H',hi,'Water')/1e3 for pi,hi in zip(pp,hh)]
                    if kind == 'T':
                        yy = [CP.PropsSI('T','P',pi,'H',hi,'Water')-273.15 for pi,hi in zip(pp,hh)]
                    else:
                        yy = hh/1e3
                    ax.plot(ss,yy,'k--' if (a,b_) in dashed else 'k-',lw=1.3)
                for a,b_ in segs_line:
                    if kind == 'T':
                        ya,yb = self.T[a-1]-273.15,self.T[b_-1]-273.15
                    else:
                        ya,yb = self.h[a-1]/1e3,self.h[b_-1]/1e3
                    ax.plot([self.s[a-1]/1e3,self.s[b_-1]/1e3],[ya,yb],
                            'k--' if (a,b_) in dashed else 'k-',lw=1.3)
                for i in range(1,19):
                    yi = self.T[i-1]-273.15 if kind == 'T' else self.h[i-1]/1e3
                    ax.plot(self.s[i-1]/1e3,yi,'ro',ms=4)
                    ax.annotate(str(i),(self.s[i-1]/1e3,yi),textcoords='offset points',
                                xytext=(4,4),fontsize=8)
                ax.set_xlabel('s [kJ/kg/K]')
                ax.grid(True)

            # --- T-s diagram
            self.fig_Ts,ax = plt.subplots(figsize=(8,6))
            ax.plot(s_liq,Td-273.15,'b',lw=0.8)
            ax.plot(s_vap,Td-273.15,'b',lw=0.8)
            draw(ax,'T')
            ax.set_ylabel('T [°C]')
            ax.set_title('T-s diagram')

            # --- h-s diagram
            self.fig_hs,ax = plt.subplots(figsize=(8,6))
            ax.plot(s_liq,h_liq,'b',lw=0.8)
            ax.plot(s_vap,h_vap,'b',lw=0.8)
            draw(ax,'h')
            ax.set_ylabel('h [kJ/kg]')
            ax.set_title('h-s diagram')

            plt.show()

            # self.FIG = self.fig_pie_en, self.fig_pie_ex, self.fig_Ts, self.fig_hs