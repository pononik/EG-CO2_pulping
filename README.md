# EG-CO2_pulping
Conceptual EG–CO₂ pulping model with LCI for 1 t unbleached pulp; EF v3.1 LCA, OVAT, and Monte Carlo.

EG–CO₂ pulping has emerged as a promising high‑yield alternative to kraft pulping, but its environmental feasibility requires evaluation. This repository provides a conceptual process model and life cycle assessment (LCA) for EG–CO₂ pulping, including OVAT sensitivity and Monte Carlo uncertainty analyses, benchmarked against kraft.

## Summary

- Goal: Quantify environmental performance of EG–CO₂ pulping vs. kraft.
- Scope: 1 t unbleached pulp, cradle‑to‑gate; EF v3.1 impact assessment.
- Inventory: Built from lab data, process flow diagrams, and M&E balances.
- Analyses: OVAT sensitivity (key drivers) and Monte Carlo uncertainty.
- Key findings:
  - Climate change impacts: EG–CO₂ ≈ 0.318 t CO₂‑eq/t pulp; kraft ≈ 0.309 t CO₂‑eq/t.
  - EG–CO₂ outperforms in acidification, biogenic climate change, ecotoxicity, eutrophication, land use, material resources, ozone depletion, and photochemical oxidant formation.
  - Dominant drivers: ethylene glycol make‑up, pulp yield, liquor‑to‑wood ratio.
  - EG–CO₂ can be environmentally competitive if EG losses are minimized, yields maintained, and coproducts valorized.

## Repository contents

- `scripts_egco2/` — Process models, LCI construction, and analysis code
  - `EG_CO2_model.py` — M&E balance LCA, OVAT, and Monte Carlo.
  - 'Ponomarev_EGCO2_script_description_2026.pdf' - conceptual diagram for the description of the script.
- `CITATION.cff` — How to cite this work
- `LICENSE` — License (Unlicense by default; change if needed)

## Getting started

### Requirements
- Python >= 3.10
- Common packages: numpy, pandas, scipy, matplotlib, olca-ipc
- LCA tooling (choose what you use): OpenLCA with ecoinventV10, olca-ipc
