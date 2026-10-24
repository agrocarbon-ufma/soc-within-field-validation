# Between-system contrasts can inflate remote sensing accuracy for soil carbon: a call for within-field validation

<!-- After the GitHub release, replace XXXXXXX with the Zenodo DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Data and code for the manuscript *"Between-system contrasts can inflate remote sensing accuracy for soil carbon: a call for within-field validation"* (Oliveira et al., submitted to *Precision Agriculture*).

Authors of the dataset and code: Matheus Silva Oliveira and Gustavo André de Araújo Santos (Federal University of Maranhão, UFMA). Master's dissertation of M.S. Oliveira, Graduate Programme in Animal Science (PPGCA/UFMA).

The study measured soil organic carbon (SOC) and its particulate (POC) and mineral-associated (MAOC) fractions under native forest (NF, cerrado *sensu stricto*), an integrated crop–livestock system (ICL) and a managed pasture (MP) in Brejo, Maranhão, Brazil, and asked whether covariates from a UAV, PlanetScope and Sentinel-2 predict them within fields or only recognise the land-use system.

**AGROCARBON** – Research and Innovation Group in Climate-Smart Agriculture, Universidade Federal do Maranhão (UFMA/CCCh) · www.agrocarbon.ufma.br · agrocarbon@ufma.br

## Repository structure

```
data/
  AGROCARBON_dados_COT_COP_sensoriamento_2024-2025.xlsx   dataset (README, Points, Samples, Covariates, Dictionary)
  points.csv          60 sampling points: coordinates (WGS 84 and SIRGAS 2000 / UTM 23S) and influential-point flag
  samples.csv         360 samples (point × layer × season): SOC, POC, MAOC (g kg⁻¹), corrected and uncorrected values
  covariates.csv      UAV, PlanetScope, Sentinel-2, terrain and texture covariates per point
  dictionary.csv      definition, unit and source of every column
  s2_series_grid_10m.csv   Sentinel-2 time series (2019–2024) on a 10 m grid, used for the predicted POC map
notebooks/
  analysis.ipynb      all analyses of the article, in order:
                      1–3 SOC, POC and MAOC by system and layer (tests, figure, PCA and PERMANOVA)
                      4   correlation with all points × within systems
                      5   Random Forest and PLSR with nested leave-one-out cross-validation; SHAP; predicted
                          POC map; deeper layers; between-system variance (η²) × R²
                      6   sensitivity to the influential POC values (ICL points 6–9)
                      7   temporal validation (calibrate in the dry season, predict the wet season)
                      8   repeatability dry × wet season
                      9   sensitivity to the laboratory correction
  soc_analysis.py     functions used by the notebook
results/              tables (xlsx/csv) and figures (PNG 600 dpi and PDF) written by the notebook
```

Function names, variable names and code comments are in Portuguese. Codes in the code → codes in the article: `MN` → NF, `ILP8` → ICL, `PM` → MP; `COT` → SOC, `COP` → POC, `COM` → MAOC; covariate prefixes `DR_` UAV, `PL_` PlanetScope, `S2_` Sentinel-2 single date, `S2c_` Sentinel-2 wet-season composite, `S2s_` Sentinel-2 time series.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter lab        # run analysis.ipynb from top to bottom
```

Tested with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, statsmodels 0.15 and shap 0.54. Random seeds are fixed. A full run takes a few minutes on a laptop. The covariates were extracted with Google Earth Engine (`soc_analysis.coletar_covariaveis`, which needs an Earth Engine account); the extracted values are provided in `data/`, so Earth Engine is not needed to reproduce the results.

## Data

- 60 sampling points (20 per system) on a regular grid, located with a GNSS receiver; layers 0–100, 100–200 and 200–300 mm; dry season (2 July 2024; ICL under pearl millet) and wet season (1 April 2025; soybean at R5–R6).
- SOC by wet oxidation with potassium dichromate (Walkley & Black, 1934); POC as the fraction retained on a 53 µm sieve after dispersion (Cambardella & Elliott, 1992); MAOC = SOC − POC. Titrations standardised against the unheated blank of each batch and calibrated with a reference soil; uncorrected values and correction factors are also given.
- UAV: MicaSense RedEdge-P on a DJI Matrice 350 RTK, 120 m flight height, September 2024, Agisoft Metashape. The orthomosaics are available from the authors on request.
- PlanetScope surface reflectance (cloud-free scenes with exposed soil, April–October 2024); Sentinel-2 L2A; Copernicus GLO-30 DEM (© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA); SoilGrids 2.0. Covariates are means within a 10 m radius of each point.

## How to cite

Data and code:

Oliveira, M.S., Santos, G.A.A., 2026. Data and code for: Between-system contrasts can inflate remote sensing accuracy for soil carbon: a call for within-field validation (v1.0.0) [Dataset and software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Article:

Oliveira, M.S., et al. Between-system contrasts can inflate remote sensing accuracy for soil carbon: a call for within-field validation. *Precision Agriculture* (submitted). [year and DOI to be added after publication]

When using these data, please cite both the dataset and the article and acknowledge the AGROCARBON Research Group (UFMA). We would be grateful to receive a copy of published works that use the data (agrocarbon@ufma.br).

## Funding

CAPES (Finance Code 001); FAPEMA (CARBOSCAN, APP-05024/23); CNPq (SpectralCarbon, 405484/2023); EMAP and Port of Itaqui; Embrapa.

## License

Code: MIT (see `LICENSE`). Data: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

### Resumo em português

Dados e código do artigo sobre carbono orgânico do solo (COT) e suas frações particulada (COP) e associada aos minerais (COM) em mata nativa, ILP e pastagem manejada em Brejo-MA, com covariáveis de drone, PlanetScope e Sentinel-2. Rode `notebooks/analysis.ipynb` do início ao fim; os caminhos são relativos à pasta do repositório. Dados sob CC BY 4.0: ao usar, cite o conjunto de dados e o artigo e mencione o Grupo AGROCARBON/UFMA.
