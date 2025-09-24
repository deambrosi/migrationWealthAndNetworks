
---

## 🚀 Usage

1. **Clone repo** and open in MATLAB.
2. Run `Main.m`:
   - Initializes parameters, grids, and settings.
   - Solves steady-state equilibrium with no help (`noHelpEqm`).
   - Iterates on dynamic equilibrium via `solveDynamicEquilibrium`.
   - Simulates agent trajectories with converged policies (`simulateAgents`).
3. Results are displayed in the console; you can save outputs (value functions, policy functions, simulated paths) by uncommenting the save block at the end of `Main.m`.

---

## 🔧 Requirements

- MATLAB R2020b or later (uses `pagemtimes` for batched multiplications).
- Statistics Toolbox (for some estimation routines).
- Reasonable RAM (agent-level simulation can be large with `Nagents = 5000` and `T = 100`).

---

## 📊 Data & Estimation

- **Microdata**: Migrant surveys in Colombia (EPM), Peru (ENPOVE), Ecuador (EPEC), Chile (ENM).  
- **Macro data**: R4V stock series by destination, UNHCR protection monitoring.  
- **Estimation strategy**: Two-step GMM:
  1. **Step 1**: Estimate tenure profiles of outcomes in a “Hotel California” counterfactual (stayers only).  
  2. **Step 2**: Estimate mobility, help, and utility parameters using flows, hazards, and survey moments:contentReference[oaicite:7]{index=7}.

See `/docs/ESTIMATION_STRATEGY.txt` for details.

---

## 📝 Citation

If you use this code, please cite as:

> Deambrosi, Agustín (2025). *Dynamic Equilibrium of Migrant Location Choice: Evidence from the Venezuelan Exodus*. PhD Thesis, Penn State University.

---

## 📌 Notes

- This codebase is designed for transparency and modularity: each block (parameters, grids, transitions, simulation) is in a separate file.  
- The README corresponds to **Version 2.1 (September 2025)**, consistent with `Main.m`.  
- Model description is in `/docs/ModelDraft.txt` and `/docs/NewModelEnvironment.txt`; data background in `/docs/IntroducitonBackgroundDataDraft.txt`.

---
