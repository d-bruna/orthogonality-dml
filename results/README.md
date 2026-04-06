# Results

## dml_scan.log
Raw output from the 648-cell pure simulation. Each cell's results are logged
with the following format:

```
CELL n=<n> p=<p> tau=<tau> sp=<s/p> rho=<rho>
  Lasso: cover=<...> bias=<...> rmse=<...> sqrtNDn=<...> R2=<...>
  Ridge: cover=<...> bias=<...> rmse=<...> sqrtNDn=<...> R2=<...>
```

641 of 648 cells produced stable results. 7 cells failed at extreme
configurations (p > n with strong signals and low correlation).

## Parsing
The log can be parsed into structured JSON using the parsing code in
`code/generate_figures.py`. The parsed output (`cells.json`) contains
one dictionary per cell with fields: n, p, tau, sp, rho, lasso_cover,
ridge_cover, lasso_bias, ridge_bias, lasso_sqrtNDn, ridge_sqrtNDn,
lasso_R2, ridge_R2.
