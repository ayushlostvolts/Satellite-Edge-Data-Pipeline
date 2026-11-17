# 15-Year Climate Data Analysis & Satellite Edge Filtering Benchmark (2010 – 2024)

## Executive Summary
This directory contains the multi-decadal performance evaluation and scientific visualization of the **Onboard Satellite Edge Rainfall Processing Pipeline** applied across **15 consecutive years (2010–2024, 5,479 continuous daily grid slices)** of India Meteorological Department (IMD) 0.25° gridded daily rainfall data ($129 \times 135$ spatial matrix, over $95.4$ million individual measurements).

The study is organized into two dedicated folders:

1. [filtering_graphs/](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/): **Scientific Visualizations & Climatological Time Series**
2. [metrics_evaluation/](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/): **Comprehensive Multi-Metric Benchmarks, Statistical Tables & Charts**

---

## Folder 1: Scientific Filtering Graphs (`filtering_graphs/`)

Contains 8 high-resolution figures demonstrating the operational stability and physical preservation of the edge pipeline across 15 continuous years:

| Figure | Filename | Key Scientific Finding |
| :--- | :--- | :--- |
| **Fig 1** | [fig1_15yr_continuous_monsoon_timeseries.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig1_15yr_continuous_monsoon_timeseries.png) | 15-year continuous record (5,479 days) across 4 climatic regimes (Mumbai, Cherrapunji, Kolkata, Jaisalmer). Zero baseline drift, robust dry-season squelch. |
| **Fig 2** | [fig2_annual_peak_preservation_comparison.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig2_annual_peak_preservation_comparison.png) | Year-by-year annual maximum storm intensity ($P_{\max}$). Edge filter preserves $98.5\%$ peak ratio, while Butterworth/Median suffer 35–50% peak clipping. |
| **Fig 3** | [fig3_15yr_wavelet_multiscale_scalogram.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig3_15yr_wavelet_multiscale_scalogram.png) | Dyadic DWT subband decomposition (A3 seasonal envelope, D3 active-break, D2 synoptic, D1 convective transients). Exact Parseval energy conservation ($< 10^{-12}$). |
| **Fig 4** | [fig4_adaptive_gain_dynamics_historic_floods.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig4_adaptive_gain_dynamics_historic_floods.png) | Zooms into historic disasters (2013 Uttarakhand cloudburst, 2018 Kerala floods, 2024 monsoon). Shows $\alpha[n]$ snapping to 0.95 within 1 sample. |
| **Fig 5** | [fig5_spatial_15yr_climatology_raw_vs_filtered.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig5_spatial_15yr_climatology_raw_vs_filtered.png) | National 15-year mean annual spatial climatology heatmaps over India. Preserves orographic boundaries (Western Ghats, Northeast) without spatial blur. |
| **Fig 6** | [fig6_spatial_15yr_extreme_storm_hotspots.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig6_spatial_15yr_extreme_storm_hotspots.png) | 15-year spatial map of extreme rain days ($R_{95p}$) and detected CEPS priority flash-flood triggers. |
| **Fig 7** | [fig7_15yr_cumulative_hydrological_mass_balance.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig7_15yr_cumulative_hydrological_mass_balance.png) | Cumulative mass balance curve over 5,479 days. 30-day volume preservation eliminates long-term sensor drift (bias $< 0.05\%$). |
| **Fig 8** | [fig8_15yr_spectral_welch_density.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/fig8_15yr_spectral_welch_density.png) | Welch PSD showing sharp stopband attenuation ($f > 0.35\text{ day}^{-1}$) with zero attenuation of intra-seasonal climate oscillations ($f < 0.05\text{ day}^{-1}$). |

Detailed documentation: [filtering_graphs/README_FILTERING_GRAPHS.md](file:///e:/Coding/4th%20year%20Project/15_year_analysis/filtering_graphs/README_FILTERING_GRAPHS.md)

---

## Folder 2: Comprehensive Metrics & Statistical Evaluation (`metrics_evaluation/`)

Evaluates **26 individual quantitative metrics** across 7 methods (Satellite Edge Filter, Butterworth Lowpass 3, Median 5, Moving Average 5, Moving Average 3, Savitzky-Golay 11, and Raw Noisy Input):

- **Error Metrics**: MAE, MSE, RMSE, NRMSE (Range), NRMSE (Mean), MedAE, MaxAE, MAPE (Wet).
- **Fidelity Metrics**: $\Delta$SNR (dB), PSNR (dB), Peak Preservation Ratio (PPR), Pearson $r$, Spearman $\rho$, SSIM-1D.
- **Hydrological Skill**: Kling-Gupta Efficiency (KGE), Nash-Sutcliffe Efficiency (NSE), PBIAS (%), 30-Day Block Bias (%).
- **Categorical Storm Skill**: Accuracy, Precision, Recall/POD, Specificity, F1-Score, Critical Success Index (CSI / Threat Score), False Alarm Ratio (FAR), Heidke Skill Score (HSS), Equitable Threat Score (ETS), F1 Heavy Rain.

### Generated Chart Gallery:

| Chart | Filename | Description |
| :--- | :--- | :--- |
| **Chart 1** | [chart1_annual_error_metrics_trends.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart1_annual_error_metrics_trends.png) | 15-year annual trajectory line plots of MAE, RMSE, MedAE, NRMSE. |
| **Chart 2** | [chart2_annual_fidelity_and_snr_trends.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart2_annual_fidelity_and_snr_trends.png) | 15-year annual trends of $\Delta$SNR, PSNR, and Peak Preservation Ratio. |
| **Chart 3** | [chart3_hydrological_efficiency_kge_nse.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart3_hydrological_efficiency_kge_nse.png) | Annual comparison of KGE, NSE, and PBIAS across methods. |
| **Chart 4** | [chart4_classification_f1_csi_pod_curves.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart4_classification_f1_csi_pod_curves.png) | Categorical detection skill scores (F1, CSI, POD, FAR). |
| **Chart 5** | [chart5_multimetric_radar_spider.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart5_multimetric_radar_spider.png) | 8-axis normalized radar profile comparing all filters. |
| **Chart 6** | [chart6_boxplots_metric_distributions_15yr.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart6_boxplots_metric_distributions_15yr.png) | Statistical box & whisker distributions across all 15 years and climate zones. |
| **Chart 7** | [chart7_contingency_confusion_matrices.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart7_contingency_confusion_matrices.png) | Rain/no-rain confusion matrix heatmaps. |
| **Chart 8** | [chart8_taylor_diagram_15yr.png](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/chart8_taylor_diagram_15yr.png) | Taylor diagram showing correlation, variance, and centered RMSE. |

### Data Files:
- [metrics_15yr_annual_detailed.csv](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/metrics_15yr_annual_detailed.csv) — 420 individual evaluated rows (15 years $\times$ 4 stations $\times$ 7 methods $\times$ 26 metrics).
- [metrics_15yr_summary_averages.csv](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/metrics_15yr_summary_averages.csv) — 15-year mean $\pm$ standard deviation for every metric.
- [README_METRICS.md](file:///e:/Coding/4th%20year%20Project/15_year_analysis/metrics_evaluation/README_METRICS.md) — Comprehensive mathematical specification and summary tables.
