# Commodity Price Predictability Analysis

This repository contains the full analysis and models used to evaluate the predictability of prices for key global commodities, including **Gold**, **Crude Oil**, and **Coffee**. The study leverages advanced statistical and machine learning techniques to uncover trends, interdependencies, and predictive relationships in the price data.

## Table of Contents
- [Objectives](#objectives)
- [Features](#features)
- [Results](#results)
---


## Project Overview

### Objectives
- Analyze the unique characteristics of commodity price time series data.
- Develop and compare models to predict price movements.
- Provide actionable insights into the dynamics of global commodity markets.

### Features
- **Preprocessing:** Handles non-stationarity through first-order differencing.
- **Modeling Approaches:**
  - Vector Autoregressive (VAR) Model
  - Autoregressive Integrated Moving Average (ARIMA) Model
  - Seasonal ARIMA (SARIMA) Model
  - Linear Regression Analysis
- **Metrics and Evaluation:**
  - Akaike Information Criterion (AIC) for lag selection.
  - Root Mean Squared Error (RMSE) and Mean Squared Error (MSE).
  - Hypothesis and F-tests for statistical significance.

## Results

The following sections summarize the key findings and model performance:

### 1. Vector Autoregressive (VAR) Model
- Optimal lag order: **6** (determined using AIC).
- Significant interdependencies:
  - **Gold** shows a strong response to lagged prices of **Crude Oil** and **Coffee**.
  - **Coffee** exhibits mean-reverting behavior but limited influence from other commodities.
- Best suited for analyzing dynamic relationships across commodities.

### 2. ARIMA Model
- Commodity-specific models with the following configurations:
  - **Gold:** ARIMA(2, 0, 1) – High prediction error due to volatility.
  - **Crude Oil:** ARIMA(1, 0, 1) – Best performance with low error margins.
  - **Coffee:** ARIMA(1, 0, 0) – Underfitting issues with limited variability capture.
- ARIMA successfully captures linear patterns but struggles with non-linear dynamics.

### 3. Seasonal ARIMA (SARIMA) Model
- Seasonal orders: (1, 1, 1, 12) for all commodities.
- Results:
  - **Crude Oil:** Closely matches actual prices with accurate seasonal forecasts.
  - **Gold:** Struggles with non-linear volatility, leading to larger prediction intervals.
  - **Coffee:** Captures general trends but fails during high variability.

### 4. Linear Regression Analysis
- Key predictors include:
  - Macroeconomic indicators (e.g., VIX, Dollar Index, S&P500, Federal Rates).
  - Lagged prices of other commodities.
- **R² Scores:**
  - Gold: **0.741**
  - Crude Oil: **0.587**
  - Coffee: **0.492**
- High explanatory power for Gold, moderate for Crude Oil, and limited for Coffee.

---
