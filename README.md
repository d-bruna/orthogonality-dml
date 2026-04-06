# When Does Orthogonality Fail to Protect?

## Ridge versus Lasso as Nuisance Estimators in Double Machine Learning

**Authors:** Daniel Garcia Bruña, Renske Gosselaar, Johnny Hu, Emile Moyson  
**Supervisor:** Anastasija Tetereva  
**Programme:** Seminar in Machine Learning, Erasmus School of Economics  
**Date:** April 2026

---

## Abstract

Double Machine Learning (DML) protects causal estimates from first-stage estimation errors through Neyman's orthogonality condition. We show that this protection is a **finite insurance mechanism**, not an unconditional guarantee. Under dense weak signals — the regime characterising most economic data — Lasso-based DML coverage drops to **35%** while Ridge-based DML maintains **69%**. The mechanism: Lasso's thresholding discards genuine but individually weak coefficients, inflating the nuisance error product beyond the orthogonality budget. We propose a simple diagnostic: comparing out-of-fold R² from both methods on each nuisance regression identifies the researcher's exposure.

## Key Findings

| Finding | Detail |
|---------|--------|
| Coverage breakdown | Lasso 35% vs Ridge 69% at τ=0.1 (R²≈9%) |
| More data makes it worse | Gap widens from 9pp (n=300) to 34pp (n=1000) |
| Asymmetric protection | Breakdown requires *both* equations dense+weak |
| Ridge failure | 91pp gap favouring Lasso under sparse signals |
| Cross-dataset consistency | L/R ratio stable at 1.41–1.61 across all datasets |

## Repository Structure

```
├── paper/
│   ├── paper_final.tex          # Complete LaTeX manuscript
│   ├── MyReferencesFile.bib     # Bibliography
│   └── figuresfinal/            # All 13 figures (PDF, vector)
│       ├── fig1.pdf             # Coverage vs τ, three sample sizes
│       ├── fig2.pdf             # Mechanism: √n·Δn and coverage
│       ├── fig3.pdf             # R² comparison
│       ├── fig4.pdf             # Four-panel parameter sensitivity
│       ├── fig5.pdf             # p/n × ρ interaction
│       ├── fig6.pdf             # Breakdown point, three p/n
│       ├── fig7.pdf             # Gap asymmetry
│       ├── fig8.pdf             # Ridge failure in sparse regime
│       ├── fig9.pdf             # FRED-MD √n·Δn
│       ├── fig10.pdf            # FRED-MD coverage
│       ├── fig11.pdf            # Barro-Lee two-panel
│       ├── fig12.pdf            # Cross-dataset comparison
│       └── fig13_cauchy_schwarz.pdf  # C-S tightness (supplementary)
│
├── code/
│   ├── simulation.py            # 648-cell pure simulation (main)
│   ├── semi_synthetic.py        # FRED-MD semi-synthetic (33 cells)
│   ├── semi_synthetic_barrolee.py  # Barro-Lee semi-synthetic (11 cells)
│   ├── asymmetric_dml.py        # Asymmetric nuisance designs
│   ├── empirical.py             # Empirical diagnostics
│   └── generate_figures.py      # Figure generation script
│
├── data/
│   ├── fred_md.csv              # FRED-MD predictor matrix
│   └── barrolee.csv             # Barro-Lee growth dataset
│
├── results/
│   └── dml_scan.log             # Raw 648-cell simulation output
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Compilation

### Paper
```bash
cd paper/
pdflatex paper_final.tex
bibtex paper_final
pdflatex paper_final.tex
pdflatex paper_final.tex
```

Requires: `eur.png` (Erasmus logo) in the `paper/` directory. The paper uses standard LaTeX packages available in TeX Live 2023+. Optionally uncomment `lmodern` and `microtype` in the preamble for improved typography.

### Simulation
```bash
pip install numpy scipy scikit-learn pandas joblib matplotlib
cd code/
python simulation.py          # ~63 hours on 8-core M-series Mac
python semi_synthetic.py       # ~60 minutes
python semi_synthetic_barrolee.py  # ~10 minutes
```

## Simulation Design

**648-cell grid:**
- Signal strength: τ ∈ {5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01}
- Sparsity: s/p ∈ {0.05, 0.30, 1.0}
- Dimensionality: p ∈ {100, 200, 500}
- Sample size: n ∈ {300, 500, 1000}
- Correlation: ρ ∈ {0.0, 0.5, 0.8}

B=100 replications per cell, S=3 cross-fitting repetitions, K=5 folds.

**Semi-synthetic:**
- FRED-MD: n ∈ {757, 400, 250}, p=122, 33 cells
- Barro-Lee: n=90, p=61, 11 cells

## The Diagnostic (Practitioner Summary)

Before committing to a nuisance learner in DML:

1. **Run both Lasso and Ridge** on each nuisance regression
2. **Compare out-of-fold R²** separately for outcome (ℓ₀) and treatment (m₀)
3. **Interpret:**
   - Ridge R² >> Lasso R² on **both** → dense weak signals, use Ridge
   - Gap on **one** regression only → asymmetric, partial protection
   - Similar R² → choice doesn't matter

## Citation

```bibtex
@unpublished{garciabruna2026orthogonality,
  title   = {When Does Orthogonality Fail to Protect? {R}idge versus
             {L}asso as Nuisance Estimators in Double Machine Learning},
  author  = {Garcia Bru{\~n}a, Daniel and Gosselaar, Renske and
             Hu, Johnny and Moyson, Emile},
  year    = {2026},
  note    = {MSc Thesis, Erasmus School of Economics},
}
```

## Runtime

| Component | Time | Hardware |
|-----------|------|----------|
| Pure simulation (648 cells) | 3,804 min | 8-core Apple M-series |
| FRED-MD semi-synthetic (33 cells) | 60 min | Same |
| Barro-Lee semi-synthetic (11 cells) | 10 min | Same |
| Figure generation | 2 min | Same |

## License

This repository accompanies an academic thesis. Code is provided for reproducibility. Please cite the paper if you use any part of this work.
