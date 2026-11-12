import os
import numpy as np
import matplotlib.pyplot as plt
from steam_turbine_group_02 import steam_turbine


BASE = dict(
    p_6=65e5,
    p_8=160e5,
    p_9=40e5,
    p_ref=1e5,
    T_ref=288.15,
    T_max=833.15,  # 560 °C
    T_drum=425,
    T_cd_out=298.15,
    T_pinch_cd=11,
    T_pinch_ex=5,
    x_min=0.88,
    eta_is_HP=0.90,
    eta_is_LP=0.88,
    eta_mec=0.98,
    eta_pump=0.85,
    k_heat=0.95,
)

P_E = 500e6

# None -> p_9 stays fixed at BASE['p_9'] (40 bar)
# 0.25 -> p_9 = 25 % of p_8 (reheat pressure follows p_8, closer to real design)
P9_RATIO = None


NAN = dict(eta_en=np.nan, eta_ex=np.nan, x11=np.nan,
           mdot=np.nan, feasible=False)


def run(**over):
    st = steam_turbine(P_E, {**BASE, **over}, False)
    st.evaluate()
    return st


def kpis(st):
    # Physically consistent extraction pressures: p11 < p16 < p_drum < p14
    ordering = st.p_11 < st.p_16 < st.p_4 < st.p_14
    # Positive extraction fractions and flow
    positive = min(st.X_1, st.X_2, st.X_3) > 0 and st.m_dot_cond > 0
    if not (np.isfinite(st.eta_cyclen) and ordering and positive):
        return NAN

    return dict(
        eta_en=st.eta_cyclen,
        eta_ex=st.eta_cyclex,
        x11=st.x_11,
        mdot=st.m_dot_tot,
        feasible=st.x_11 >= BASE['x_min'],   # constraint on the LP exit quality
    )


def sweep(p8_values, **fixed):
    out = []
    for p8 in p8_values:
        over = dict(p_8=p8, **fixed)
        if P9_RATIO is not None:
            over['p_9'] = P9_RATIO * p8
        try:
            out.append(kpis(run(**over)))
        except Exception:
            out.append(NAN)
    return out


get = lambda res, k: np.array([r[k] for r in res], dtype=float)


# ============================================================
# Variation de p_8, pour plusieurs températures maximales
# ============================================================

# p_8 must stay above p_6 (65 bar), otherwise pump P3 has nothing to do
p8_values = np.linspace(100e5, 400e5, 40)         # 100 -> 400 bar

Tmax_list = [773.15, 813.15, 833.15, 873.15, 923.15]   # 500 -> 650 °C
results = {Tm: sweep(p8_values, T_max=Tm) for Tm in Tmax_list}


# ============================================================
# Plots
# ============================================================

plots = [
    ('eta_en', 100, 'Cycle energy efficiency [%]', r'HW4 - Steam turbine-20261105\plots\eta_en_vs_p8.png'),
    ('eta_ex', 100, 'Cycle exergy efficiency [%]', r'HW4 - Steam turbine-20261105\plots\eta_ex_vs_p8.png'),
    ('x11',    1,   'Vapour quality at LP exit $x_{11}$ [-]', r'HW4 - Steam turbine-20261105\plots\x11_vs_p8.png'),
    ('mdot',   1,   'Total steam mass flow [kg/s]', r'HW4 - Steam turbine-20261105\plots\mdot_vs_p8.png'),
]

for key, scale, ylabel, fname in plots:
    fig, ax = plt.subplots(figsize=(7, 5))
    for Tm in Tmax_list:
        res = results[Tm]
        y = scale * get(res, key)
        ok = np.array([r['feasible'] for r in res])
        line, = ax.plot(p8_values / 1e5, y, ls=':', lw=1)                # all points
        ax.plot(p8_values / 1e5, np.where(ok, y, np.nan),                # x11 >= x_min
                color=line.get_color(), lw=2,
                label=f'$T_{{max}}$ = {Tm - 273.15:.0f} °C')
    if key == 'x11':
        ax.axhline(BASE['x_min'], color='r', ls='--', label='$x_{min}$')
    ax.axvline(220.64, color='gray', ls='-.', lw=1)                      # critical pressure
    ax.set_xlabel('Maximum steam pressure $p_8$ [bar]')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=200)

plt.show()