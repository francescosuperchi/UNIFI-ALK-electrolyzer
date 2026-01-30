
"""

MIT License

Copyright (c) 2026 UNIFI Wind Energy

Created: 2026-01-30

"""
import matplotlib.pyplot as plt

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math
import copy

from UNIFI_alk_model_electrochemical import _solve_power_to_current_density, BaseElectroCell, ElectroCellALK
from UNIFI_alk_model_thermal import BankThermalALK, thermal_step, V_TN_CELL


#%%
'model behavior plot - efficiency vs load at different temperatures'

_PLOT_CURRENT_POINTS = 200

def _cell_efficiency_vs_load(cell: BaseElectroCell) -> Tuple[np.ndarray, np.ndarray]:
    cell.build_curves()
    current_density = cell.arCurrentDensity
    current = current_density * cell.rA_cell
    faraday_eff = cell.faraday_efficiency(current_density)

    rLHV_H2 = 119_988.0  # J/g
    rMu = 2.01588        # g/mol
    rN = 2.0
    rLossDry = 0.03
    rConstantPart = rMu / cell.rF / rN * (1.0 - rLossDry)

    h2_dot = faraday_eff * rConstantPart * current
    p_total = current * cell.arV_cell
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = (rLHV_H2 * h2_dot) / p_total
    efficiency = np.nan_to_num(efficiency, nan=0.0, posinf=0.0, neginf=0.0)

    rated_power = max(float(p_total[-1]), 1e-6)
    load_pct = np.clip(p_total / rated_power, 0.0, None) * 100.0
    return load_pct, efficiency * 100.0

def plot_alk_efficiency_vs_load(
    ax=None,
    temps_C: Optional[np.ndarray] = None,
    title: str = "ALK Efficiency vs Load",
    xlim: Optional[Tuple[float, float]] = (0.0, 100.0),
    ylim: Optional[Tuple[float, float]] = (0.0, 70.0),
    min_temp: float = 40.0,
    max_temp: float = 80.0,
    n_temps: int = 15,
):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if temps_C is None:
        temps_C = np.linspace(min_temp, max_temp, n_temps)
    else:
        temps_C = np.asarray(temps_C, dtype=float).ravel()
        if temps_C.size == 0:
            temps_C = np.linspace(min_temp, max_temp, n_temps)
        min_temp = float(np.min(temps_C))
        max_temp = float(np.max(temps_C))

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=min_temp, vmax=max_temp)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for T in temps_C:
        cell = ElectroCellALK()
        cell.rT = float(T)
        cell.iNumCurrent = _PLOT_CURRENT_POINTS
        load_pct, efficiency_pct = _cell_efficiency_vs_load(cell)
        color_val = cmap(norm(T))
        ax.plot(load_pct, efficiency_pct, color=color_val, linewidth=1.5, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Load [% of rated power]")
    ax.set_ylabel("Efficiency [% LHV]")
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Temperature [C]")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    return ax
ax = plot_alk_efficiency_vs_load()
ax.figure.tight_layout()
plt.show()


#%%
'activation example: triangular power input'

def run_triangular_power_example(
    n_cells: int,
    n_stacks: int,
    ramp_time_s: float = 1800.0,
    dt: float = 1.0,
    T_amb: float = 25.0,
    min_power_frac: float = 0.20,
) -> Dict[str, np.ndarray]:
    cell = ElectroCellALK()
    th = BankThermalALK(n_cells=n_cells, n_stacks=n_stacks)

    cell.rT = th.rT
    cell.build_curves()
    p_cell_curve = cell.arCurrentDensity * cell.rA_cell * cell.arV_cell
    p_cell_max = float(np.max(p_cell_curve))
    n_total_cells = int(th.n_cells * th.n_stacks)
    p_total_max = p_cell_max * n_total_cells
    min_power_frac = float(np.clip(min_power_frac, 0.0, 1.0))
    min_power_W = p_total_max * min_power_frac

    n_up = max(int(ramp_time_s / dt), 1)
    n_down = n_up
    p_profile = np.concatenate(
        [
            np.linspace(0.0, p_total_max, n_up + 1, endpoint=True)[:-1],
            np.linspace(p_total_max, 0.0, n_down + 1, endpoint=True),
        ]
    )
    times = np.arange(p_profile.size) * dt
    p_delivered = np.where(p_profile >= min_power_W, p_profile, 0.0)

    rMu = 2.01588        # g/mol
    rN = 2.0
    rLossDry = 0.03
    rConstantPart = rMu / cell.rF / rN * (1.0 - rLossDry)

    temps = []
    h2_cum_g = []
    p_cool_elec = []
    h2_mass_g = 0.0

    for p_total in p_delivered:
        p_cell = p_total / max(float(n_total_cells), 1.0)
        cell.rT = th.rT
        j, v_cell = _solve_power_to_current_density(cell, p_cell)
        current = j * cell.rA_cell

        faraday_eff = float(cell.faraday_efficiency(np.array([j]))[0])
        h2_dot_cell = faraday_eff * rConstantPart * current
        h2_dot_total = h2_dot_cell * n_total_cells
        h2_mass_g += h2_dot_total * dt

        q_cell = current * max(v_cell - V_TN_CELL, 0.0)
        q_total = q_cell * n_total_cells
        diag = {}
        th = thermal_step(th, q_total, T_amb, dt, out=diag)

        temps.append(th.rT)
        h2_cum_g.append(h2_mass_g)
        p_cool_elec.append(diag.get("p_cool_elec", 0.0))

    return {
        "time_s": times,
        "p_in_W": p_profile,
        "p_delivered_W": p_delivered,
        "T_C": np.asarray(temps, dtype=float),
        "h2_cum_g": np.asarray(h2_cum_g, dtype=float),
        "p_cool_elec_W": np.asarray(p_cool_elec, dtype=float),
        "p_min_W": np.array([min_power_W], dtype=float),
        "p_peak_W": np.array([p_total_max], dtype=float),
    }

#define the number of cells and number of stacks composing the bank of the electorlyzer
N_CELLS = 10
N_STACKS = 9

result = run_triangular_power_example(n_cells=N_CELLS, n_stacks=N_STACKS)
total_h2_kg = result["h2_cum_g"][-1] / 1000.0
duration_h = result["time_s"][-1] / 3600.0 if result["time_s"].size > 1 else 0.0
peak_power_kw = result["p_peak_W"][0] / 1000.0
final_temp_c = result["T_C"][-1]
max_temp_c = float(result["T_C"].max())

'output print'
print("Triangular power example")
print(f"Duration: {duration_h:.2f} h")
print(f"Peak power: {peak_power_kw:.2f} kW (total)")
print(f"Total H2 produced: {total_h2_kg:.4f} kg")
print(f"Final temperature: {final_temp_c:.2f} C")
print(f"Max temperature: {max_temp_c:.2f} C")

'time plot'
t_hours = result["time_s"] / 3600.0
p_kw = result["p_in_W"] / 1000.0
p_delivered_kw = result["p_delivered_W"] / 1000.0
h2_kg = result["h2_cum_g"] / 1000.0
p_cool_kw = result["p_cool_elec_W"] / 1000.0

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
axes[0].plot(t_hours, p_kw, color="tab:blue", label="Available power")
axes[0].plot(t_hours, p_delivered_kw, color="tab:gray", linestyle="--", label="Delivered power")
axes[0].set_ylabel("Power [kW]")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="best")

axes[1].plot(t_hours, h2_kg, color="tab:green")
axes[1].set_ylabel("H2 produced [kg]")
axes[1].grid(alpha=0.3)

axes[2].plot(t_hours, result["T_C"], color="tab:red")
axes[2].set_ylabel("Temperature [C]")
axes[2].grid(alpha=0.3)

axes[3].plot(t_hours, p_cool_kw, color="tab:orange")
axes[3].set_ylabel("Cooling power required [kW]")
axes[3].set_xlabel("Time [h]")
axes[3].grid(alpha=0.3)

fig.suptitle("Triangular power example")
fig.tight_layout()
plt.show()
  
