# Data Sources

## fred_md.csv
FRED-MD monthly macroeconomic database (McCracken and Ng, 2016).
- Source: https://research.stlouisfed.org/econ/mccracken/fred-databases/
- 126 series, January 1959 – January 2026
- After preprocessing: n=757, p=122
- See Appendix B of the paper for transformation details

## barrolee.csv
Barro-Lee cross-country growth dataset (Barro and Lee, 1994).
- Source: `hdm` R package
- n=90 countries, p=61 predictors + outcome
- Variables: GDP, education, demographics, government, trade, political stability

## Notes
- Both datasets are standardised to column mean 0 and variance 1 before use
- FRED-MD subsamples (n=400, n=250) drawn with fixed random seed (42)
