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
HW4 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HW4)

from steam_turbine_group_xx import steam_turbine

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

my_ST = steam_turbine(inputs, params, True)
my_ST.evaluate()

eta_en = my_ST.eta_cyclen
