"""
LELME2150 - Thermal cycles
Homework 4 - Steam turbine

Test code for your function

@author: Zoé Dassesse
@date: October 2026
"""

#
#===IMPORT PACKAGES============================================================
#

import os
import sys
import matplotlib.pyplot as plt
HW4 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HW4)

from steam_turbine_group_02 import steam_turbine

p_6             = 65e+5
p_8             = 160e+5
p_9             = 40e+5
p_ref, T_ref    = 1e+5,   288.15
T_max           = 833.15
T_drum          = 425
T_cd_out        = 298.15
T_pinch_cd      = 11
T_pinch_ex      = 5
x_min           = 0.88
eta_is_HP       = 0.90
eta_is_LP       = 0.88
eta_mec         = 0.98
eta_pump        = 0.85
k_heat          = 0.95
P_e             = 500e+6

inputs = P_e
params = {'p_6':          p_6,
          'p_8':          p_8,
          'p_9':          p_9,
          'p_ref':        p_ref,
          'T_ref':        T_ref,
          'T_max':        T_max,
          'T_drum':       T_drum,
          'T_cd_out':     T_cd_out,
          'T_pinch_cd':   T_pinch_cd,
          'T_pinch_ex':   T_pinch_ex,
          'x_min':        x_min,
          'eta_is_HP':    eta_is_HP,
          'eta_is_LP':    eta_is_LP,
          'eta_mec':      eta_mec,
          'eta_pump':     eta_pump,
          'k_heat':       k_heat
          }

print_states = True
my_ST = steam_turbine(inputs, params, True)
my_ST.evaluate()
print(f"Flows rate : m_dot_tot = {my_ST.m_dot_tot:.3f} kg/s | m_dot_cond = {my_ST.m_dot_cond:.3f} kg/s")
print(f"Efficiencies : eta_cyclen = {my_ST.eta_cyclen:.3f} | eta_cyclex = {my_ST.eta_cyclex:.3f} | eta_rotex = {my_ST.eta_rotex:.3f}")
print(f"Losses : loss_mec = {my_ST.loss_mec/1e3:.3f} kW | loss_conden = {my_ST.loss_conden/1e3:.3f} kW | loss_rotex = {my_ST.loss_rotex/1e3:.3f} kW | loss_condex = {my_ST.loss_condex/1e3:.3f} kW | loss_valvex = {my_ST.loss_valvex/1e3:.3f} kW | loss_rest = {my_ST.loss_transex/1e3:.3f} kW")
print(f"Flows : X1 = {my_ST.X_1:.3f} | X2 = {my_ST.X_2:.3f} | X3 = {my_ST.X_3:.3f}\n")
print(f"{'State':>5} {'p [bar]':>12} {'T [K]':>12} {'h [kJ/kg]':>12} {'s [kJ/kgK]':>12} {'x [-]':>10} {'e [kJ/kg]':>12}")
print("-" * 81)
for i in range(18):
    p = my_ST.p[i]
    T = my_ST.T[i]
    h = my_ST.h[i]
    s = my_ST.s[i]
    x = my_ST.x[i]
    e = my_ST.e[i]

    print(f"{i+1:>5} {p/1e5:>12.4f} {T:>12.2f} "
          f"{h/1e3:>12.2f} {s/1e3:>12.4f} "
          f"{x:>10.3f} {e/1e3:>12.2f}")
print(" Closure :", my_ST.closure)

# Show charts
plt.show()

eta_en = my_ST.eta_cyclen
