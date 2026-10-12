import numpy as np
import matplotlib.pyplot as plt
from gas_turbine_group_02 import gas_turbine

BASE = dict(T_3=1673.15, r_c=18, eta_pi_c=0.9, eta_pi_t=0.9, k_cc=0.95, k_mec=0.015,
            air=['N2','O2'], air_prop=[0.79,0.21], alkane=[1,4])
P_E = 230e6
NAN = dict(eta_cyc=np.nan, eta_ex=np.nan, w=np.nan, co2=np.nan)

def run(p_1=1e5, T_1=288.15, **over):
    gt = gas_turbine((p_1, T_1, P_E), {**BASE, **over}, False)
    gt.evaluate()
    return gt

def kpis(gt):
    # point non physique : pas assez d'air pour brûler (lambda < 1) ou travail net négatif
    if not np.isfinite(gt.excess_air) or gt.excess_air < 1 or gt.W_mcy <= 0 or gt.dotm_a <= 0:
        return NAN
    x, y = gt.x, gt.y
    co2 = gt.dotm_f*(x*44.01/(x*12.011 + y*1.008))*3.6e9/P_E          # [g CO2 / kWh_e]
    return dict(eta_cyc=gt.eta_cyclen, eta_ex=gt.eta_totex, w=gt.W_mcy/1e3, co2=co2)

def sweep(values, **fixed):
    out = []
    for v in values:
        try:
            out.append(kpis(run(r_c=v, **fixed)))
        except Exception:
            out.append(NAN)
    return out

get = lambda res, k: np.array([r[k] for r in res])

# r_c de 1 à 100 : on démarre à 1.5, car pour r_c <= 1/k_cc (≈ 1.05) la turbine ne détend pas
# et pour r_c = 1 la moyenne de gamma sur [T1, T2] devient 0/0
rc = np.linspace(1.5, 100, 60)
T3_list = [1273.15, 1473.15, 1673.15, 1873.15]
results = {T3: sweep(rc, T_3=T3) for T3 in T3_list}

plots = [
    ('eta_cyc', 100, 'Cycle energy efficiency [%]',          'cycle_energy_efficiency.png'),
    ('eta_ex',  100, 'Total exergy efficiency [%]',          'total_exergy_efficiency.png'),
    ('w',       1,   'Net specific work [kJ/kg of air]',     'net_specific_work.png'),
    ('co2',     1,   'CO2 emissions [g/kWh$_e$]',            'co2_emissions.png'),
]

for key, scale, ylabel, fname in plots:
    fig, ax = plt.subplots(figsize=(7, 5))
    for T3 in T3_list:
        ax.plot(rc, scale*get(results[T3], key), label=f'T3 = {T3-273.15:.0f} °C')
    ax.set_xlabel('Compression ratio r_c [-]')
    ax.set_ylabel(ylabel)
    ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=200)

plt.show()