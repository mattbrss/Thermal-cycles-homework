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

# Folder for the figures (created next to this script)
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

NAN = dict(eta_en=np.nan, eta_ex=np.nan, x11=np.nan,
           p1=np.nan, mdot=np.nan, feasible=False)


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
        p1=st.p_1 / 1e2,                       # condenser pressure [mbar]
        mdot=st.m_dot_tot,
        feasible=st.x_11 >= BASE['x_min'],     # constraint on the LP exit quality
    )


def sweep(Tcd_values, **fixed):
    out = []
    for Tcd in Tcd_values:
        try:
            out.append(kpis(run(T_cd_out=Tcd, **fixed)))
        except Exception:
            out.append(NAN)
    return out


get = lambda res, k: np.array([r[k] for r in res], dtype=float)


# ============================================================
# Variation de T_cd_out, pour plusieurs pressions de resurchauffe
# ============================================================

Tcd_values = np.linspace(278.15, 308.15, 40)      # 5 -> 35 °C

p9_list = [40e5]
results = {p9: sweep(Tcd_values, p_9=p9) for p9 in p9_list}


# ============================================================
# Plots
# ============================================================

plots = [
    ('eta_en', 100, 'Cycle energy efficiency [%]', 'eta_en_vs_Tcd.png'),
    ('eta_ex', 100, 'Cycle exergy efficiency [%]', 'eta_ex_vs_Tcd.png'),
    ('x11',    1,   'Vapour quality at LP exit $x_{11}$ [-]', 'x11_vs_Tcd.png'),
    ('p1',     1,   'Condenser pressure $p_1$ [mbar]', 'p1_vs_Tcd.png'),
]

for key, scale, ylabel, fname in plots:
    fig, ax = plt.subplots(figsize=(7, 5))
    for p9 in p9_list:
        res = results[p9]
        y = scale * get(res, key)
        ok = np.array([r['feasible'] for r in res])
        line, = ax.plot(Tcd_values - 273.15, y, ls=':', lw=1)            # all points
        ax.plot(Tcd_values - 273.15, np.where(ok, y, np.nan),             # x11 >= x_min
                color=line.get_color(), lw=2, label=f'$p_9$ = {p9/1e5:.0f} bar')
    if key == 'x11':
        ax.axhline(BASE['x_min'], color='r', ls='--', label='$x_{min}$')
    if key == 'p1':
        ax.set_yscale('log')
    ax.set_xlabel('Cooling water outlet temperature $T_{cd,out}$ [°C]')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=200)

plt.show()