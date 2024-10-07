# AMF Handover Duration Anomaly Detection

This repository contains two notebooks, demonstrating different approaches to anomaly detection and time series forecasting on AMF Handover duration data.

## Table of Contents
- [Notebooks](#notebooks)
  - [amf-handover-timeseries.ipynb](#amf-handover-timeseriesipynb)
  - [Granite-TS-Smoke-Test.ipynb](#granite-ts-smoke-testipynb)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Visualization](#visualization)
- [Granite Time Series Model](#granite-time-series-model)
- [Running the Notebooks](#running-the-notebooks)

## Notebooks

1. **amf-handover-timeseries.ipynb**  
   This notebook focuses on anomaly detection for AMF Handover Duration and includes the following key sections:
   - **Anomaly Detection**: Simulates high-latency anomalies during handovers.
   - **Data Source**: AMF handover data captured from a 5G emulator.
   - **Visualization**: Plots cumulative time spent on AMF handovers for anomaly detection.

2. **Granite-TS-Smoke-Test.ipynb**  
   This notebook demonstrates the usage of the `TinyTimeMixer` model for time series forecasting.
   - **Zero-shot and Few-shot Learning**: The model is evaluated directly and with limited fine-tuning.
   - **Time-Series Preprocessing**: Prepares data for forecasting using the `tsfm_public` library.
   - **Visualization**: Compares predicted vs actual values.

## Installation

To run these notebooks, the following dependencies are required:

```bash
pip install pandas torch transformers matplotlib
```

Additionally, the `tsfm_public` library must be installed for time-series forecasting:

```bash
pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.8"
```

## Data Sources
- **AMF Handover Data**: Captured from a 5G simulator using Prometheus metrics. The dataset is located at:  
  `open-digital-platform-2_0/5G_Emulator_API/core_network/amf-handover-request-durations.csv`.

## Visualization

Both notebooks include visualizations to aid in understanding the AMF handover durations and predictions. 

Example of plotting cumulative handover duration:

```python
import matplotlib.pyplot as plt
plt.plot(df.index, df["Cumulative_Sum_Value"])
plt.show()
```

## Granite Time Series Model

The Granite Time Series Model utilizes `TinyTimeMixer` for forecasting. Pre-trained models are fetched from [Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1).

## Running the Notebooks

1. **Anomaly Detection (amf-handover-timeseries.ipynb)**:
   - Run the notebook to simulate anomalies and visualize results.

2. **Granite Time Series Forecasting (Granite-TS-Smoke-Test.ipynb)**:
   - Use this notebook to predict future AMF handover durations using the `TinyTimeMixer` model.
