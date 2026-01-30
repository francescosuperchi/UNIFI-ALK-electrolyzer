
"""

MIT License

Copyright (c) 2026 UNIFI Wind Energy

Created: 2026-01-30

"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math
import copy

V_TN_CELL = 1.48  # [V] thermoneutral per cell (use a better T/P law if desired)

# thermal model based on https://doi.org/10.1016/j.renene.2023.03.077

@dataclass
class BaseBankThermal(ABC):
    n_cells: int
    n_stacks: int
    
    k_gen = 1.0

    # geometry/walls used by thermal_step
    s1: float = 0.004
    r3: float = 1.0
    T_nominal: float = 60.0

    # convection/conduction params
    h1: float = 0.0
    h2: float = 10.0
    h3: float = 20.0
    k1: float = 52.0

    # insulation toggle (can change anytime)
    insulated: bool = False

    # nominal insulation options
    _s2_unins: float = 1e-4
    _k2_unins: float = 52.0
    _s2_ins: float  = 0.2
    _k2_ins: float  = 0.05

    # effective values computed on the fly (no staleness)
    @property
    def s2(self) -> float:
        return self._s2_ins if self.insulated else self._s2_unins

    @property
    def k2(self) -> float:
        return self._k2_ins if self.insulated else self._k2_unins

@dataclass
class BankThermalALK(BaseBankThermal):
    n_cells_design: int = 100    #number of cells in the reference device
    L_design: float = 3.0        #lenght of the reference device
    r1_design: float = 0.3       #radius of the reference device

    h1: float = 100.0

    L_stack: float = field(default=3.0, init=False)
    _r1_direct: float = field(default=0.3, init=False)

    c_elect: float = 900.0
    density_elect: float = 1000.0

    T_nominal: float = 70.0
    T_min: float = 20.0
    T_max: float = 71.0
    rT: float = 65.0

    def __post_init__(self):
        scale = max(float(self.n_cells) / float(self.n_cells_design), 1e-9) ** (1.0 / 3.0)
        self.L_stack = self.L_design * scale
        self._r1_direct = self.r1_design * scale

    def _equiv_radius_r1(self) -> float:
        return self._r1_direct
    
def thermal_step(
                    th: BaseBankThermal,
                    Q_el: float,
                    T_amb: float,
                    dt: float,
                    cop_cooling: float = 4.0,
                    clamp: bool = True,
                    out: Optional[dict] = None,
                ) -> BaseBankThermal:
    
    th_new = copy.deepcopy(th)

    # Geometry  (see scheme in fig. 3 https://doi.org/10.1016/j.renene.2023.03.077)
    L  = th_new.L_stack * th_new.n_stacks
    r1 = th_new._equiv_radius_r1()
    r2 = r1 + th_new.s1
    r3 = th_new.r3
    r4 = th_new.r3 + th_new.s2

    # Conductance chain (all linear in ΔT)
    
    #internal water convection
    a = th_new.h1 * 2.0 * math.pi * r1 * L   
    # inner wall conduction
    b = th_new.k1 * 2.0 * math.pi * L / max(math.log(r2 / r1),1e-12)
    
    c = th_new.h2 * 2.0 * math.pi * r2 * L
    # air-gap convection proxy
    d = th_new.h2 * 2.0 * math.pi * r3 * L
    # insulation conduction
    e = th_new.k2 * 2.0 * math.pi * L / max(math.log(r4 / r3),1e-12)
    # outer convection
    f = th_new.h3 * 2.0 * math.pi * r4 * L
    
    values = [a, b, c, d, e, f]
    G_eq = 1.0 / sum((1/x if x != 0 else 0) for x in values)

    # Heat balance terms
    q_lost = G_eq * (th_new.rT - T_amb)                 # [W]
    
    q_gain = th_new.k_gen * max(Q_el, 0.0)              # heat generation as a fraction of the power losses Q_el [W]
    
    T_nom = getattr(th_new, "T_nominal", th_new.rT)
    q_lost_nom = G_eq * max(T_nom - T_amb, 0.0)
    if th_new.rT >= T_nom:
        q_cool = max(q_gain - q_lost_nom, 0.0)
    else:
        q_cool = 0.0
    p_cool = q_cool / cop_cooling if cop_cooling > 0 else 0.0

    # half-filled gas-liquid separator (water + KOH)
    V = L * (r1 ** 2) * math.pi
    m = V * th_new.density_elect * 0.5
    C_th = m * th_new.c_elect

    dT = dt * (q_gain - q_lost - q_cool) / C_th if C_th > 0.0 else 0.0
    
    T_new = th_new.rT + dT
    if clamp:
        T_new = float(np.clip(T_new, th_new.T_min, th_new.T_max))
    th_new.rT = T_new

    # Fill diagnostics (if requested)
    if out is not None:
        out["q_gain"] = q_gain
        out["q_lost"] = q_lost
        out["q_cool"] = q_cool
        out["p_cool_elec"] = p_cool
        out["G_eq"]   = G_eq
        out["C_th"]   = C_th

    return th_new
