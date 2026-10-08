"""
LELME2150 - Thermal cycles
Homework 2 - Gas turbines

Test code for your function

"""

#
#===IMPORT PACKAGES============================================================
#

from gas_turbine_group_02 import gas_turbine
import matplotlib.pyplot as plt

p_1, T_1 = 1e+5, 288.15 # [Pa], [K]
r_c = 18
T_3 = 1673.15 # [K]
eta_pi = 0.9 # [-]
k_mec = 0.015 # [-]
k_cc = 0.95
P_e = 230e+6

inputs = p_1,T_1,P_e
params =  {'T_3':       T_3,
           'r_c':       r_c,
           'eta_pi_c':  eta_pi,
           'eta_pi_t':  eta_pi,
           'k_cc':      k_cc,
           'k_mec':     k_mec,
           'air':       ['N2','O2'],
           'air_prop':  [0.79,0.21],
           'alkane':    [1,4]
           }
my_GT = gas_turbine(inputs,params,True)
my_GT.evaluate()

print("total gas prop: (should be 1.0)",sum(my_GT.gas_prop))
print("Cycle data:")
print("Pressure:",my_GT.p)
print("Temperature:",my_GT.T)
print("Enthalpy:",my_GT.h)
print("Entropy:",my_GT.s)
print("Exergy:",my_GT.e)
print("Combustion data:")
print("LHV",my_GT.LHV)
print("Excess air",my_GT.excess_air)
print("Gas composition:",my_GT.gas,my_GT.gas_prop)
print("Mass flow data:")
print("Air mass flow rate",my_GT.dotm_a)
print("Fuel mass flow rate",my_GT.dotm_f)
print("Gas mass flow rate",my_GT.dotm_g)
print("Energy efficiencies:")
print("Cycle energy efficiency",my_GT.eta_cyclen)
print("Total energy efficiency",my_GT.eta_toten)
print("Mechanical efficiency",my_GT.eta_mec)
print("Exergy efficiencies:")
print("Cycle exergy efficiency",my_GT.eta_cyclex)
print("Total exergy efficiency",my_GT.eta_totex)
print("Rotor exergy efficiency",my_GT.eta_rotex)
print("Combustion exergy efficiency",my_GT.eta_combex)
print("Energy losses")
print("Mechanical losses",my_GT.loss_mec)
print("Exhaust energy losses",my_GT.loss_echen)
print("Exergy losses")
print("Mechanical losses",my_GT.loss_mec)
print("Rotor exergy losses",my_GT.loss_rotex)
print("Combustion exergy losses",my_GT.loss_combex)
print("Exhaust exergy losses",my_GT.loss_echex)
print("Cycle work",my_GT.WORK)

print("Exergy power",my_GT.dotm_a*my_GT.e_1,my_GT.dotm_a*my_GT.e_2,my_GT.dotm_g*my_GT.e_3,my_GT.dotm_g*my_GT.e_4)
plt.show()