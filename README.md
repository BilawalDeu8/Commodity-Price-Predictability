# Commodity Price Predictability Analysis

This repository contains the full analysis and models used to evaluate the predictability of prices for key global commodities, including **Gold**, **Crude Oil**, and **Coffee**. The study leverages advanced statistical and machine learning techniques to uncover trends, interdependencies, and predictive relationships in the price data.

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

### Key Insights
- VAR modeling reveals significant inter-commodity relationships.
- ARIMA and SARIMA highlight temporal and seasonal price patterns.
- Linear regression identifies macroeconomic and market predictors impacting commodity prices.
- Gold prices are highly responsive to other commodities, while Coffee demonstrates exogenous behavior.

---
