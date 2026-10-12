
import numpy as np
import matplotlib.pyplot as plt

from gas_turbine_group_02 import gas_turbine


# ============================================================
# Paramètres fixes de la turbine
# ============================================================

BASE = dict(
    T_3=1673.15,
    r_c=18,
    eta_pi_c=0.9,
    eta_pi_t=0.9,
    k_cc=0.95,
    k_mec=0.015,
    air=["N2", "O2"],
    air_prop=[0.79, 0.21],
)

P_E = 230e6  # Puissance électrique demandée [W]

# Combustibles disponibles dans gas_turbine_group_02.py
FUELS = {
    "Méthane": [1, 4],
    "Propane": [3, 8],
    "Hydrogène": [0, 2],
}

NAN = {
    "eta_cyc": np.nan,
    "eta_ex": np.nan,
    "w": np.nan,
    "co2": np.nan,
}


# ============================================================
# Simulation
# ============================================================

def run(alkane):
    parameters = {
        **BASE,
        "alkane": alkane,
    }

    gt = gas_turbine(
        (1e5, 288.15, P_E),
        parameters,
        False,
    )
    gt.evaluate()
    return gt


def kpis(gt):
    # Rejeter les points physiquement invalides
    if (
        not np.isfinite(gt.excess_air)
        or gt.excess_air < 1
        or gt.W_mcy <= 0
        or gt.dotm_a <= 0
        or gt.dotm_f < 0
    ):
        return NAN.copy()

    x, y = gt.x, gt.y

    # Fraction massique de carbone dans le combustible
    M_f = x * 12.011 + y * 1.008

    if M_f <= 0:
        return NAN.copy()

    # Émissions directes de CO2 [g/kWh électrique]
    co2 = (
        gt.dotm_f
        * (x * 44.01 / M_f)
        * 3.6e9
        / P_E
    )

    return {
        "eta_cyc": gt.eta_cyclen,
        "eta_ex": gt.eta_totex,
        "w": gt.W_mcy / 1e3,  # [kJ/kg d'air]
        "co2": co2,           # [g/kWh]
    }


# ============================================================
# Calcul des résultats pour chaque combustible
# ============================================================

results = {}

for name, composition in FUELS.items():
    try:
        gt = run(composition)
        results[name] = kpis(gt)

        print(
            f"{name:10s} | "
            f"excès d'air = {gt.excess_air:.3f} | "
            f"rendement cycle = {gt.eta_cyclen:.3f} | "
            f"rendement exergétique total = {gt.eta_totex:.3f} | "
            f"consommation de carburant = {gt.dotm_f:.3f} kg/s | "
            f"travail spécifique net = {gt.W_mcy/1e3:.3f} kJ/kg d'air | "
            f"émissions directes de CO2 = {gt.dotm_f * (composition[0] * 44.01 / (composition[0] * 12.011 + composition[1] * 1.008)) * 3.6e9 / P_E:.3f} g/kWh"

        )

    except Exception as error:
        results[name] = NAN.copy()
        print(f"{name} : simulation échouée ({error})")


# ============================================================
# Tracé des 4 graphes
# ============================================================

plots = [
    ("eta_cyc", "Rendement énergétique du cycle", "[-]", 100),
    ("eta_ex", "Rendement exergétique total", "[-]", 100),
    ("w", "Travail spécifique net", "[kJ/kg d'air]", 1),
    ("co2", "Émissions directes de CO2", "[g/kWh]", 1),
]

for key, title, unit, scale in plots:
    names = list(results.keys())
    values = np.array([
        results[name][key] * scale for name in names
    ])

    fig, ax = plt.subplots(figsize=(8, 5))

    valid = np.isfinite(values)
    ax.bar(
        np.array(names)[valid],
        values[valid],
    )

    if not valid.any():
        ax.text(
            0.5, 0.5,
            "Aucun résultat valide",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(title)
    ax.set_ylabel(
        f"{unit} " + ("[%]" if scale == 100 else "")
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()

plt.show()