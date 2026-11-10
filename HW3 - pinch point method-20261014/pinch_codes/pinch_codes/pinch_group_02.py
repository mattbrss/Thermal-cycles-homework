"""
LELME2150 - Thermal cycles
Homework 3.0 - Pinch model heat exchanger

Signature of the function for the heat exchanger pinch model.

@author: Mattéo Hauglustaine
@date: October 10th, 2025
"""

#
# ===IMPORT PACKAGES============================================================

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

#
# ===HEAT EXCHANGER MODEL - TO BE IMPLEMENTED==========================================
#


class heat_exchanger(object):
    """
    Class for the simulation of heat exchangers

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
        -> Compute the heat exchange
    """

    def __init__(self, inputs, parameters, display):
        """
        Create a heat exchanger object.
        """
        p_hs, T_hs_su, T_hs_ex, T_cs_su, T_cs_ex = inputs
        self.p_hs = p_hs  # [Pa] hot side fluid pressure
        self.T_hs_su = T_hs_su  # [K]  hot side fluid supply temperature
        self.T_hs_ex = T_hs_ex  # [K]  hot side fluid exit temperature
        self.T_cs_su = T_cs_su  # [K]  cold side fluid supply temperature
        self.T_cs_ex = T_cs_ex  # [K]  cold side fluid exit temperature
        self.pinch_target = parameters["pinch_target"]  # [K]  target for pinch point temperature difference
        self.fluid_hs = parameters["wf_hot_side"]  # hot side working fluid
        self.fluid_cs = parameters["wf_cold_side"]  # cold side working fluid
        self.p_cs_guess = parameters["p_cs_guess"]  # cold side working fluid pressure guess
        self.display = display  # if True, make the plots

    def evaluate(self):
        """
        This is the main method of the heat exchanger class.
        It finds the working fluid pressure to reach the pinch point and
        computes the exergy efficiency of the heat exchanger.
        """

        T0_C = 25 # expressed in Celcius °C
        T0 = T0_C + 273.15 # expressed in Kelvin K
        n = 300
        h_hs_su = CP.PropsSI('H','T',self.T_hs_su,'P',self.p_hs,self.fluid_hs)
        h_hs_ex = CP.PropsSI('H','T',self.T_hs_ex,'P',self.p_hs,self.fluid_hs)

        def profiles(p_evap):
            h_cs_su = CP.PropsSI('H','T',self.T_cs_su,'P',p_evap,self.fluid_cs)
            h_cs_ex = CP.PropsSI('H','T',self.T_cs_ex,'P',p_evap,self.fluid_cs)
            hl = CP.PropsSI('H','P',p_evap,'Q',0,self.fluid_cs)
            hv = CP.PropsSI('H','P',p_evap,'Q',1,self.fluid_cs)
            # determining h_cold
            h_cold = np.linspace(h_cs_su,h_cs_ex,n)
            h_cold = np.unique(np.concatenate([h_cold,[hl,hv]]))
            h_cold = h_cold[(h_cold >= h_cs_su) & (h_cold <= h_cs_ex)]

            x = (h_cold - h_cs_su)/(h_cs_ex-h_cs_su)
            h_hot = h_hs_ex + x*(h_hs_su-h_hs_ex)
            T_cold = np.array([CP.PropsSI('T','H',h,'P',p_evap,self.fluid_cs) for h in h_cold])
            T_hot = np.array([CP.PropsSI('T','H',h,'P',self.p_hs,self.fluid_hs) for h in h_hot])

            return x,h_cs_su,h_cs_ex,T_hot,T_cold

        def residual(p_bar):
            p_evap = p_bar[0]*1e5
            _,_,_,T_hot,T_cold = profiles(p_evap)
            return np.min(T_hot-T_cold)-self.pinch_target #bc we search the min value between both curves

        #solution of the explicit equation
        sol = fsolve(residual,self.p_cs_guess/1e5,xtol=1e-8)
        # =========================================================================================
        # with the AI help : to verify if there is a convergence or not
        # sol, info, ier, msg = fsolve(residual, self.p_cs_guess/1e5, xtol=1e-8, full_output=True)
        # if ier != 1:
        #     print(f"Warning: fsolve did not converge ({msg})")
        # =========================================================================================
        self.p_evap_solution = sol[0]*1e5
        x,h_cs_su,h_cs_ex,T_hot,T_cold = profiles(self.p_evap_solution)
        self.measured_pinch = round(np.min(T_hot-T_cold),4)
        self.x, self.T_hot, self.T_cold = x, T_hot, T_cold

        #entropy
        p = self.p_evap_solution
        s_cs_su = CP.PropsSI('S','T',self.T_cs_su,'P',p,self.fluid_cs)
        s_cs_ex = CP.PropsSI('S','T',self.T_cs_ex,'P',p,self.fluid_cs)
        s_hs_su = CP.PropsSI('S','T',self.T_hs_su,'P',self.p_hs,self.fluid_hs)
        s_hs_ex = CP.PropsSI('S','T',self.T_hs_ex,'P',self.p_hs,self.fluid_hs)

        #heat and exergy
        # by doing an energy balance we have (m_c)*(h_cs_ex-h_cs_su) = (m_h)*(h_hs_su-h_hs_ex)
        # we obtain : (m_h)/(m_c) = m_ratio = (h_cs_ex-h_cs_su)/(h_hs_su-h_hs_ex) = qtot/(h_hs_su-h_hs_ex) because Q = Delta h
        # m_v (e3-e2) / m_f (ef - eexh)
        qtot = h_cs_ex-h_cs_su
        m_ratio = qtot/(h_hs_su-h_hs_ex) # [kg_oil / kg_heptane]
        diff_e_c = qtot - T0*(s_cs_ex-s_cs_su) # [J/kg_heptane]
        diff_e_h = m_ratio*((h_hs_su-h_hs_ex) - T0*(s_hs_su-s_hs_ex)) #multiplied by m_ratio because incorrect units without that
        self.eta_transex = diff_e_c/diff_e_h
        self.irreversibility = diff_e_h-diff_e_c

        print(f"Optimal evaporation pressure: {self.p_evap_solution/1e5:.2f} bar")
        print(f"Pinch point value measured: {self.measured_pinch} K")
        print(f"Exergy efficiency of HEX: {self.eta_transex:.2%}")

        print('========================================')
        print("value of m_ratio :",round(m_ratio,3))
        print("values enthalpies :", round((h_hs_su-h_hs_ex)/1000,3), "kJ/kg") 
        print("irreversible losses :",self.irreversibility) #quantity that we can't recover
        print('========================================')
        # print figures in celcius
        if self.display:
            q_kW = x*qtot / 1e3  # [kJ/kg_wf]
            plt.figure(figsize=(8,6))
            plt.plot(q_kW,T_hot-273.15,"r",label="Therminol 72 (hot)")
            plt.plot(q_kW,T_cold-273.15,"b",label="Heptane (cold)")
            i = np.argmin(T_hot-T_cold)
            plt.plot([q_kW[i]]*2, [T_cold[i]-273.15,T_hot[i]-273.15],
                     "k--", label=f"Pinch = {self.measured_pinch:.2f} K")
            plt.xlabel("Heat transferred q [kJ/kg of heptane]")
            plt.ylabel("Temperature [°C]")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # =========
        # notes
        # =========

        # if we vary the value of the reference temperature T0, we obtain a different value for the efficiency
        # (if T0 increases then eta_transex decreases, and vice-versa, the rest stays the same (pressure and desired pinch)
        # we can also confirm that the pressure is always 4.68 bars, and the pinch point value is always 5K (as asked)
        # and an efficiency of eta_transex = 89.63% 
        #we can also say that the pinch could be at another point, such as the one at te right instead of where it already is
        #(so where the temprature of therminol72 is at its maximum)

# tests for different values of the pinch
if __name__ == "__main__":
    PINCHS = [-14.3, -5, 2, 3, 5, 10, 20]  # [K] values to compare

    inputs = (2e5, 220 + 273.15, 150 + 273.15, 125 + 273.15, 200 + 273.15)
    p_guess = 5e5  # [Pa]
    results = []

    fig, ax = plt.subplots(figsize=(8, 6))
    for dT in PINCHS:
        params = {
            "pinch_target": dT,
            "wf_hot_side": "INCOMP::T72",
            "wf_cold_side": "Heptane",
            "p_cs_guess": p_guess,
        }
        hx = heat_exchanger(inputs, params, False)  # False: no individual figure
        hx.evaluate()
        p_guess = hx.p_evap_solution  # start from the previous solution
        results.append((dT, hx.p_evap_solution / 1e5, hx.measured_pinch, hx.eta_transex))
        ax.plot(
            hx.x,
            hx.T_cold - 273.15,
            label=f"Heptane, pinch {dT} K ({hx.p_evap_solution/1e5:.2f} bar)",
        )

    ax.plot(hx.x, hx.T_hot - 273.15, "r", lw=2, label="Therminol 72")
    ax.set_xlabel("Fraction of heat exchanged x [-]")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    # plt.savefig("Tq_pinch_sensitivity.png", dpi=200)  # uncomment for the report
    plt.show()

    print("\n=========== Sensitivity summary ===========")
    print(f"{'pinch [K]':>10} {'p [bar]':>10} {'measured [K]':>14} {'eta_transex':>12}")
    for dT, p_bar, pinch_meas, eta in results:
        print(f"{dT:>10} {p_bar:>10.3f} {pinch_meas:>14.3f} {eta:>12.2%}")