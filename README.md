# ALK-electrolyzer-model_UNIFI
Main repository for a physics-based, parametric model of an alkaline (ALK) water electrolyzer cell developed by the University of Florence (Italy).
This repository provides a lightweight Python implementation to generate ALK cell performance curves and efficiency trends as a function of operating conditions. The model combines standard electrochemical relationships (reversible voltage, activation losses, ohmic losses) with literature-based correlations and a temperature-dependent Faraday efficiency fit. References are embedded directly in the source code as DOI links in the comments.

---

## What is included

- **Physics-based ALK cell model (ElectroCellALK)**

	- Polarization curve generation (current density range up to rated value)

	- Reversible voltage in standard conditions plus Nernst correction (pressure and vapor terms)

	- Activation overpotentials via Tafel formulation (anode and cathode)

	- Ohmic losses including electrode contributions and an empirical resistance term

	- Faraday efficiency correlation with optional temperature-dependent coefficient interpolation

- **Utility functions**

	- Computation of LHV efficiency vs load (based on electrical power and H₂ production rate)

- **Example plotting script**

	- Efficiency vs load across a temperature sweep (color-mapped by temperature)

## Repository structure (typical)

- alk_cell_model.py (or equivalent module containing the classes and example script)

- Optional: examples/ and docs/ (if added later)

## Requirements

- Python 3.x

- numpy

- matplotlib (only required to run the plotting example)

## Quick start (typical usage)

- Instantiate the cell (ElectroCellALK)

- Set operating parameters (e.g., rT, pressure_bar, koh_mass_frac, geometry)

- Call build_curves() to populate arrays such as:

- arCurrentDensity (A/cm²)

- arV_cell (V)

- arE_min (V, reversible component)

- arR_cell (effective resistance terms)

- Optionally evaluate Faraday efficiency with faraday_efficiency(J)

## Notes

- The implementation is cell-level and intended for parametric studies, system integration, and control-oriented performance maps.

- Default parameters are “representative values” as indicated in the code; for design-grade studies, parameters should be replaced with stack-specific data and validated against experiments or vendor curves.

## Design updates and contributions

Design updates are welcome via Pull Requests or by contacting the authors. When contributing, it is recommended to:

- document new correlations and assumptions in code comments (with DOI/reference),

- include a minimal test or example that reproduces expected trends.

## Citation

If this model is used in research or publications, please cite the associated technical report or software record (to be added). If no report is available yet, consider citing the repository itself (release tag and commit hash) and the primary sources referenced in the code comments.
