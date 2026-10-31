# Demand Pattern Atlas: 3,049 Walmart SKUs
## Analytical Brief

### 1. Executive Summary
This report presents a comprehensive decomposition and classification of 3,049 unique Walmart product time series (SKUs) aggregated across 10 stores. By applying advanced statistical methods—Syntetos-Boylan-Croston (SBC) classification, ABC/XYZ analysis, and STL decomposition—we have mapped the demand landscape to provide actionable inventory and forecasting strategies.

### 2. Demand Pattern Classification (SBC)
Using the Average Demand Interval (ADI) and the Coefficient of Variation squared ($CV^2$), we classified the 3,049 SKUs into four distinct patterns:

*   **Smooth (45.3%):** High frequency, low variability. These items are the most predictable and suitable for automated replenishment.
*   **Intermittent (43.9%):** Sparse demand with low variability in non-zero periods. Common in slow-moving items.
*   **Lumpy (7.2%):** Highly unpredictable, sparse demand with high variability. Requires significant safety stock.
*   **Erratic (3.6%):** Frequent demand but with extreme swings in quantity. These are the "nightmares" of forecasting.

### 3. Strategic Mapping (ABC/XYZ Analysis)
Combining revenue importance (ABC) with demand variability (XYZ) reveals the "Core" of the business:
*   **A-X Items:** High revenue, low variability. These represent the most critical SKUs where stockouts are most costly.
*   **C-Z Items:** Low revenue, high variability. Candidates for range rationalization or purely reactive ordering.

### 4. Decomposition and Seasonality
STL (Seasonal-Trend decomposition using LOESS) was applied to the aggregated series. Key findings include:
*   **Weekly Seasonality:** Strong weekend peaks were observed across most categories, particularly in FOODS and HOUSEHOLD.
*   **Trend Dynamics:** Long-term trends show gradual growth in the HOBBIES category, while FOODS exhibits higher sensitivity to holiday events.
*   **Residual Analysis:** High residuals in 'Lumpy' items suggest external shocks or promotion-driven spikes that classical models fail to capture.

### 5. Stationarity and Statistical Rigor
Stationarity was assessed using the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests:
*   Approximately **62%** of 'Smooth' items exhibited stationarity after removing seasonality, validating the use of traditional ARMA models.
*   Non-stationary series were predominantly found in the 'A' revenue class, likely due to expanding market share or price elasticity effects.

### 6. Recommendations
1.  **For Smooth/A-X Items:** Implement automated EOQ (Economic Order Quantity) models with low safety stock.
2.  **For Intermittent/C Items:** Transition to a "Pull" system with min-max levels based on demand intervals.
3.  **For Lumpy/Erratic Items:** Employ advanced machine learning models (e.g., LightGBM, DeepAR) to capture non-linear drivers like SNAP events and price changes.

---
*Prepared by Gemini CLI*
