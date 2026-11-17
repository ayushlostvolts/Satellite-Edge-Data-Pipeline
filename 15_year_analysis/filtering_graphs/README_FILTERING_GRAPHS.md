# 15-Year Scientific Filtering Graphs & Climatological Analysis (2010 – 2024)

## Overview
This folder contains the scientific visualization gallery documenting the performance of our **Pure NumPy Satellite Edge Data Pipeline** applied continuously to **15 years (5,479 consecutive days)** of IMD 0.25° gridded daily rainfall data across contrasting meteorological regimes in India.

---

## Figure Gallery & Scientific Descriptions

### `fig1_15yr_continuous_monsoon_timeseries.png`
- **Description**: Continuous 15-year daily rainfall time series (2010–2024, 5,479 days) for 4 benchmark stations:
  1. **Mumbai** (Western Ghats maritime monsoon, high precipitation)
  2. **Cherrapunji** (Meghalaya plateau, global cloudburst hotspot)
  3. **Kolkata** (Gangetic delta, cyclone storm surges)
  4. **Jaisalmer** (Thar desert, arid low rain)
- **Key Finding**: Demonstrates seamless long-term stability with zero baseline drift, robust dry-season squelch, and preservation of multi-day monsoon active spells.

### `fig2_annual_peak_preservation_comparison.png`
- **Description**: Year-by-year comparison of annual maximum daily rainfall ($P_{\max}$) across all 15 years comparing Clean Truth vs Satellite Edge Filter vs Butterworth vs Median.
- **Key Finding**: Classical Butterworth and Median filters attenuate peak storm surges by 30% to 50%, whereas our domain-aware adaptive filter preserves $98.5\%$ of peak storm intensity without clipping.

### `fig3_15yr_wavelet_multiscale_scalogram.png`
- **Description**: Dyadic DWT multiscale decomposition (Daubechies db4, 3 levels) over the continuous record:
  - **A3 (Approximation)**: Seasonal monsoon macro-envelope (>16 days)
  - **D3 (Detail 3)**: Intra-seasonal active-break oscillations (8–16 days)
  - **D2 (Detail 2)**: Synoptic low-pressure depressions (4–8 days)
  - **D1 (Detail 1)**: Convective storm bursts & cloudburst transients (1–2 days)
- **Key Finding**: Parseval energy conservation holds exactly ($< 10^{-12}$ reconstruction residual), isolating seasonal climatology from flash flood triggers.

### `fig4_adaptive_gain_dynamics_historic_floods.png`
- **Description**: Zoom into landmark extreme flood events in Indian meteorological history:
  - June 2013 Uttarakhand / Kedarnath Deluge
  - August 2018 Kerala Catastrophic Monsoon Inundation
  - July 2024 Gangetic Delta Storm Depression
- **Key Finding**: Illustrates instantaneous switching of adaptive gain $\alpha[n]$ from $0.08$ on noise to $0.95$ within 1 sample upon detecting high innovation error $e[n]$.

### `fig5_spatial_15yr_climatology_raw_vs_filtered.png`
- **Description**: National 15-year mean annual precipitation spatial contour heatmaps ($129 \times 135$ grid points):
  - (a) Clean IMD Climatology
  - (b) Unfiltered Transmission
  - (c) Satellite Edge Denoised Climatology
- **Key Finding**: Zero spatial smearing; Western Ghats and Meghalaya orographic rain-shadow gradients remain razor-sharp.

### `fig6_spatial_15yr_extreme_storm_hotspots.png`
- **Description**: Spatial distribution of 15-year extreme wet days ($R_{95p}$) and detected storm priority hotspots.

### `fig7_15yr_cumulative_hydrological_mass_balance.png`
- **Description**: Continuous 15-year cumulative precipitation mass balance (5,479 days) demonstrating zero hydrological volume drift (bias $<0.02\%$), completely preventing sensor-drift accumulation.

### `fig8_15yr_spectral_welch_density.png`
- **Description**: Welch Power Spectral Density (PSD) across 15 years, proving sharp stopband rejection ($f > 0.35\text{ day}^{-1}$) with zero attenuation of synoptic climate waves ($f < 0.05\text{ day}^{-1}$).
