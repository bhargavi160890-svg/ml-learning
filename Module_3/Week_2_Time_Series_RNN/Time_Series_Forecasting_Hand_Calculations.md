# Time Series Forecasting — Hand-Calculation Practice (Wind Turbine Sensor Data)

Practice dataset: hourly **wind speed readings (m/s)** from a turbine sensor.

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Speed (m/s) | 5.2 | 5.8 | 6.1 | 5.9 | 6.4 | 6.7 | 6.3 | 6.9 | 7.1 | 6.8 |

Goal: forecast the wind speed at **t = 11**, using each method by hand, so you
build intuition before trusting a library function.

---

## 1. Naive Method — use only the last point

**Idea**: tomorrow's wind speed will be the same as today's. Simplest possible
forecast — ignores everything except the most recent observation.

**Formula**:
$$
\hat{Y}_{t+1} = Y_t
$$

**By hand**:
$$
\hat{Y}_{11} = Y_{10} = 6.8
$$

**When it's reasonable**: very short-term forecasts (next few minutes) where
wind speed doesn't change drastically, or as a baseline to compare smarter
methods against — if your fancy model can't beat naive, it's not adding value.

---

## 2. Drift Method ("hike" — using the last 2 points)

**Idea**: instead of assuming no change (naive), assume the trend between the
last two points **continues** — literally "hiking" the value up or down by the
same amount as the most recent change.

**Formula** (using the two most recent points):
$$
\hat{Y}_{t+1} = Y_t + (Y_t - Y_{t-1})
$$

**By hand**:
$$
\hat{Y}_{11} = Y_{10} + (Y_{10} - Y_9) = 6.8 + (6.8 - 7.1) = 6.8 - 0.3 = 6.5
$$

Notice: wind speed had just **dropped** from 7.1 → 6.8, so the drift method
assumes it will keep dropping by that same amount, forecasting **6.5**.

**General drift formula** (textbook version, using the *first* and *last*
points across the whole series, for forecasting `h` steps ahead):
$$
\hat{Y}_{t+h} = Y_t + h \cdot \frac{Y_t - Y_1}{t - 1}
$$

**By hand** (h=1, using first point Y₁=5.2 and last point Y₁₀=6.8):
$$
\hat{Y}_{11} = 6.8 + 1 \cdot \frac{6.8 - 5.2}{10 - 1} = 6.8 + \frac{1.6}{9} = 6.8 + 0.178 = 6.978
$$

Both versions are valid "drift" — the 2-point version reacts fast to recent
swings; the full-series version smooths the trend over the whole dataset. Try
both by hand and compare which matches your intuition about the data.

---

## 3. Full Average Method — use every point equally

**Idea**: ignore recency entirely — just average everything ever observed.

**Formula**:
$$
\hat{Y}_{t+1} = \frac{1}{t}\sum_{i=1}^{t} Y_i
$$

**By hand**:
$$
\hat{Y}_{11} = \frac{5.2+5.8+6.1+5.9+6.4+6.7+6.3+6.9+7.1+6.8}{10} = \frac{63.2}{10} = 6.32
$$

**Limitation to notice**: wind speed has clearly been *trending upward*
(5.2 → 6.8 over time), but the full average (6.32) sits well below the most
recent values (~6.8–7.1) — because it gives equal weight to old, outdated
readings. This is why full average is rarely used alone for trending data
like wind speed; it reacts far too slowly.

---

## 4. Moving Average — average of a fixed recent window

**Idea**: a compromise between naive (too reactive) and full average (too
slow) — only average the last `k` points.

**Formula** (window size k):
$$
\hat{Y}_{t+1} = \frac{1}{k}\sum_{i=t-k+1}^{t} Y_i
$$

**By hand, window = 3** (average of last 3 points: 6.9, 7.1, 6.8):
$$
\hat{Y}_{11} = \frac{6.9+7.1+6.8}{3} = \frac{20.8}{3} = 6.933
$$

**By hand, window = 5** (average of last 5 points: 6.7, 6.3, 6.9, 7.1, 6.8):
$$
\hat{Y}_{11} = \frac{6.7+6.3+6.9+7.1+6.8}{5} = \frac{33.8}{5} = 6.76
$$

**Notice the effect of window size**:
- Small window (3) → more reactive to recent changes, closer to naive.
- Large window (5+) → smoother, closer to full average, but slower to react
  to real shifts (e.g. an incoming storm front).

For wind turbine data specifically, window size should reflect **how fast
conditions genuinely change** — very short windows may just be noise-chasing
(turbulence, sensor jitter), while very long windows might miss a real
approaching weather system.

---

## 5. Single Exponential Smoothing (SES)

**Idea**: like moving average, but instead of a hard cutoff (in/out of the
window), give **exponentially decreasing weight** to older points — the most
recent point matters most, but nothing is fully discarded.

**Formula** (recursive):
$$
S_t = \alpha Y_t + (1-\alpha) S_{t-1}, \qquad S_1 = Y_1
$$
Forecast for the next period is simply the last smoothed value:
$$
\hat{Y}_{t+1} = S_t
$$

`α` (alpha, between 0 and 1) controls responsiveness:
- **High α (close to 1)** → reacts fast to recent changes (behaves more like naive).
- **Low α (close to 0)** → reacts slowly, very smooth (behaves more like full average).

**By hand, α = 0.3**:

| t | Y (actual) | S (smoothed) — calculation |
|---|---|---|
| 1 | 5.2 | S₁ = Y₁ = **5.2** (starting point, no smoothing yet) |
| 2 | 5.8 | S₂ = 0.3(5.8) + 0.7(5.2) = 1.74 + 3.64 = **5.38** |
| 3 | 6.1 | S₃ = 0.3(6.1) + 0.7(5.38) = 1.83 + 3.766 = **5.596** |
| 4 | 5.9 | S₄ = 0.3(5.9) + 0.7(5.596) = 1.77 + 3.917 = **5.687** |
| 5 | 6.4 | S₅ = 0.3(6.4) + 0.7(5.687) = 1.92 + 3.981 = **5.901** |
| 6 | 6.7 | S₆ = 0.3(6.7) + 0.7(5.901) = 2.01 + 4.131 = **6.141** |
| 7 | 6.3 | S₇ = 0.3(6.3) + 0.7(6.141) = 1.89 + 4.299 = **6.189** |
| 8 | 6.9 | S₈ = 0.3(6.9) + 0.7(6.189) = 2.07 + 4.332 = **6.402** |
| 9 | 7.1 | S₉ = 0.3(7.1) + 0.7(6.402) = 2.13 + 4.481 = **6.611** |
| 10 | 6.8 | S₁₀ = 0.3(6.8) + 0.7(6.611) = 2.04 + 4.628 = **6.668** |

**Forecast for t=11**:
$$
\hat{Y}_{11} = S_{10} = 6.668
$$

**Practice tip**: redo this table by hand with `α = 0.7` instead — you'll see
the smoothed values track the actual data much more closely (more reactive),
converging toward the naive forecast as α → 1.

---

## 6. Side-by-Side Comparison — All 5 Forecasts for t = 11

| Method | Forecast for t=11 | Reacts to recent trend? |
|---|---|---|
| Naive | 6.80 | Only the very last point |
| Drift (2-point) | 6.50 | Yes — but only the last 2 points, sensitive to noise |
| Drift (full-series) | 6.978 | Yes — trend averaged over whole series, smoother |
| Full average | 6.32 | No — treats all history equally, lags behind trend |
| Moving average (k=3) | 6.933 | Yes — recent window only |
| Moving average (k=5) | 6.76 | Somewhat — wider window, smoother |
| SES (α=0.3) | 6.668 | Gradually — exponentially decaying memory |

Actual wind speed has been oscillating upward with some noise (6.3, 6.9, 7.1,
6.8) — notice how naive and moving-average(k=3) land closest to the most
recent values, while full average badly underestimates due to unweighted old
data. This is the core lesson: **the right method depends on how much the
underlying process is trending vs. just noisy/stationary.**

---

## 7. How to Practice Further

1. **Redo every calculation above by hand with a different window/α**: try
   moving average with k=2, k=7; try SES with α=0.1 and α=0.9. Compare how
   the forecast shifts — this builds real intuition for the reactivity trade-off.

2. **Use your own wind turbine dataset** (or any real one): search for public
   wind turbine SCADA datasets (e.g. Kaggle has several "wind turbine power
   generation" datasets with wind speed, direction, and power output columns).
   Take just 10–15 rows, and manually apply each formula as done above before
   ever touching a library.

3. **Verify by hand vs. code**: once you trust your hand calculations, verify
   with `pandas`:
   ```python
   import pandas as pd
   df = pd.Series([5.2, 5.8, 6.1, 5.9, 6.4, 6.7, 6.3, 6.9, 7.1, 6.8])

   naive_forecast = df.iloc[-1]
   moving_avg_3 = df.rolling(window=3).mean().iloc[-1]
   full_avg = df.mean()
   ses = df.ewm(alpha=0.3, adjust=False).mean().iloc[-1]
   ```
   `df.ewm(alpha=..., adjust=False)` implements exactly the recursive SES
   formula used above — matching your hand calculation of 6.668.

4. **Plot the actual vs. forecasted values** for each method across the whole
   series (not just the next point) — this visually shows lag: full average
   and low-α SES will visibly lag behind real trend changes, which matters a
   lot for something like wind speed where sudden gusts/lulls are common.

5. **Extend to error metrics** — once comfortable with the hand calculations,
   learn to compute **Mean Absolute Error (MAE)** or **Root Mean Squared Error
   (RMSE)** by hand for a few points, comparing each method's forecast to the
   actual next value. This is how you'd objectively decide which method fits
   your specific turbine's wind pattern best, rather than guessing.

6. **Good learning resources**:
   - *"Forecasting: Principles and Practice"* by Hyndman & Athanasopoulos —
     free online, the standard beginner-to-intermediate reference for exactly
     these methods (naive, drift, moving average, exponential smoothing), with
     more worked examples.
   - Look for **wind power forecasting** case studies specifically — SES and
     moving averages are commonly used as simple baselines before moving to
     ARIMA or LSTM-based models for wind turbine power prediction.

---

## 8. One More Thing Worth Noticing for Wind Sensor Data Specifically

Real turbine sensor data often has **noise from turbulence** (rapid small
fluctuations) layered on top of a genuine **slower trend** (an approaching
weather front). This is exactly why:
- **Naive/drift** are too noise-sensitive alone (they'll overreact to a single
  turbulent gust).
- **Full average** is too trend-blind (it'll never "see" an approaching storm
  building over hours).
- **Moving average / SES** sit in between, and picking the right window/α is
  really about matching the smoothing to the *timescale of turbulence* you
  want to filter out, versus the *timescale of trend* you want to keep.

This is the practical reasoning wind-energy engineers use when tuning these
exact simple methods before reaching for more complex models.

---

## 9. Double Exponential Smoothing (Holt's Linear Trend Method)

**Idea**: Single exponential smoothing (SES) only tracks a smoothed *level* —
it can't project a trend, so it lags badly on data that's steadily rising or
falling (like our wind speed series). Holt's method fixes this by tracking
**two** smoothed components: a level and a trend, updated separately.

**Formulas**:
$$
L_t = \alpha Y_t + (1-\alpha)(L_{t-1} + T_{t-1}) \quad \text{(level)}
$$
$$
T_t = \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1} \quad \text{(trend)}
$$
$$
\hat{Y}_{t+h} = L_t + h \cdot T_t \quad \text{(forecast, h steps ahead)}
$$

**Initialization** (common convention): `L₁ = Y₁`, `T₁ = Y₂ - Y₁`.

**By hand, on our same wind speed series, α = 0.3, β = 0.2**:

| t | Y | Level (L) | Trend (T) |
|---|---|---|---|
| 1 | 5.2 | 5.2000 (init) | 0.6000 (init, Y₂−Y₁) |
| 2 | 5.8 | 5.8000 | 0.6000 |
| 3 | 6.1 | 6.3100 | 0.5820 |
| 4 | 5.9 | 6.5944 | 0.5225 |
| 5 | 6.4 | 6.9018 | 0.4795 |
| 6 | 6.7 | 7.1769 | 0.4386 |
| 7 | 6.3 | 7.2208 | 0.3597 |
| 8 | 6.9 | 7.3764 | 0.3188 |
| 9 | 7.1 | 7.5166 | 0.2831 |
| 10 | 6.8 | 7.4998 | 0.2231 |

Worked calculation for t=3 (to see the mechanics): $L_3 = 0.3(6.1) + 0.7(5.8+0.6) = 1.83 + 4.48 = 6.31$, then $T_3 = 0.2(6.31-5.8) + 0.8(0.6) = 0.102 + 0.48 = 0.582$.

**Forecast for t = 11 (h=1)**:
$$
\hat{Y}_{11} = L_{10} + 1 \cdot T_{10} = 7.4998 + 0.2231 = 7.723
$$

**Forecast for t = 13 (h=3)** — note the trend keeps extrapolating linearly:
$$
\hat{Y}_{13} = 7.4998 + 3(0.2231) = 8.169
$$

**Compare to SES (6.668) from Section 5**: Holt's forecast (7.723) is
noticeably higher — because it explicitly projects the upward trend forward,
while plain SES only ever "catches up" to the level and never extrapolates
beyond it. For genuinely trending sensor data (e.g. a turbine spinning up as
a storm front approaches), Holt's method is usually the better simple choice.

---

## 10. Triple Exponential Smoothing (Holt-Winters)

**Idea**: adds a **third** smoothed component — seasonality — on top of level
and trend. Needed when data repeats a pattern at a fixed period (e.g. a daily
wind cycle: calmer at night, gustier in the afternoon).

**Formulas** (additive seasonality, period `m`):
$$
L_t = \alpha (Y_t - S_{t-m}) + (1-\alpha)(L_{t-1}+T_{t-1})
$$
$$
T_t = \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1}
$$
$$
S_t = \gamma (Y_t - L_t) + (1-\gamma) S_{t-m}
$$
$$
\hat{Y}_{t+h} = L_t + h T_t + S_{t+h-m}
$$

**By hand — new example dataset** (needed because our 10-point series has no
repeating pattern): wind speed at 4 times of day (00h, 06h, 12h, 18h), across
3 days, period `m = 4`:

| Day 1 | Day 2 | Day 3 |
|---|---|---|
| 4.0, 6.5, 7.0, 5.0 | 4.3, 6.8, 7.3, 5.3 | 4.6, 7.1, 7.6, 5.6 |

**Initialization** using the first two days:
- $A_1 = \text{avg(Day 1)} = 5.625$, $A_2 = \text{avg(Day 2)} = 5.925$
- $T_0 = (A_2 - A_1)/m = 0.3/4 = 0.075$, $L_0 = A_1 = 5.625$
- Initial seasonal indices (Day 1 minus $A_1$): $S_1{=}{-}1.625,\ S_2{=}0.875,\ S_3{=}1.375,\ S_4{=}{-}0.625$
  (00h runs well below average, noon runs well above — a typical daily wind pattern)

**Running the recursion with α=0.3, β=0.2, γ=0.3** gives (rounded):

| t | Y | Level | Trend | Seasonal |
|---|---|---|---|---|
| 1 | 4.0 | 5.6775 | 0.0705 | −1.6408 |
| 4 | 5.0 | 5.7360 | 0.0446 | −0.6583 |
| 8 | 5.3 | 6.0001 | 0.0544 | −0.6709 |
| 12 | 5.6 | 6.3073 | 0.0650 | −0.6818 |

**Forecasting the next full day (t=13 to 16, h=1..4)** — reusing the seasonal
index from the matching time-of-day one cycle back:

| Forecast | Formula | Result |
|---|---|---|
| t=13 (00h) | $6.3073 + 1(0.0650) + (-1.5751)$ | **4.797** |
| t=14 (06h) | $6.3073 + 2(0.0650) + 0.8787$ | **7.316** |
| t=15 (12h) | $6.3073 + 3(0.0650) + 1.3433$ | **7.846** |
| t=16 (18h) | $6.3073 + 4(0.0650) + (-0.6818)$ | **5.886** |

Notice the forecast shape repeats the daily up-down pattern (low at 00h,
peak at 12h, dipping by 18h) while the overall level still drifts slightly
upward — this is exactly what SES/Holt alone cannot do; only the seasonal
term lets the forecast "remember" the time-of-day pattern.

**When to reach for this**: wind turbine sites with a clear daily or seasonal
cycle (diurnal heating/cooling driving wind patterns, or seasonal wind regimes
across a year) — Holt-Winters gives you level + trend + repeating pattern in
one model, still simple enough to reason about by hand.

---

## 11. ARIMA (AutoRegressive Integrated Moving Average)

**Idea**: a more general, statistically-grounded model with three parts,
written as `ARIMA(p, d, q)`:
- **AR(p)** — AutoRegressive: forecast using a weighted combination of the
  last `p` actual values.
- **I(d)** — Integrated: apply differencing `d` times to remove trend and make
  the series stationary (roughly constant mean/variance over time).
- **MA(q)** — Moving Average (of *errors*, not raw values): forecast using a
  weighted combination of the last `q` forecast errors.

**By hand — a simple `ARIMA(1,1,0)` example** (AR(1) on first-differenced data,
no MA term), on our original wind speed series:

**Step 1 — Difference once (d=1)** to remove the trend:
$$
d_t = Y_t - Y_{t-1}
$$

| t | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| $d_t$ | 0.60 | 0.30 | −0.20 | 0.50 | 0.30 | −0.40 | 0.60 | 0.20 | −0.30 |

(Rough stationarity check: mean of first half of differences ≈ 0.30, mean of
second half ≈ 0.08 — reasonably close to a constant mean, good enough for a
teaching example; real workflows use a formal test like Augmented Dickey-Fuller.)

**Step 2 — Estimate the AR(1) coefficient φ** using the lag-1 relationship on
the differenced series (a simplified, moment-based estimate — real software
uses maximum likelihood, but this shows the mechanics):
$$
\varphi \approx \frac{\sum d_t \cdot d_{t-1}}{\sum d_{t-1}^2} = \frac{-0.13}{1.39} = -0.0935
$$

**Step 3 — Forecast the next difference**:
$$
\hat{d}_{11} = \varphi \cdot d_{10} = -0.0935 \times (-0.30) = 0.028
$$

**Step 4 — Integrate back** (undo the differencing) to get the actual forecast:
$$
\hat{Y}_{11} = Y_{10} + \hat{d}_{11} = 6.8 + 0.028 = 6.828
$$

**Reading this result**: φ came out small and negative (−0.09), meaning the
differenced series barely depends on its own previous value here — this
particular AR(1) structure isn't capturing much, which is a realistic outcome
for only 10 noisy points. In practice, you'd use `statsmodels`' `ARIMA` class
(fits via maximum likelihood, not this simplified moment estimate) and use
**ACF/PACF plots** to choose `p` and `q` rather than guessing:
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data, order=(1,1,0))
fit = model.fit()
forecast = fit.forecast(steps=1)
```

**When ARIMA is worth the extra complexity over smoothing methods**: when your
turbine data is stationary (or can be made so by differencing) but doesn't
follow a simple exponential-smoothing shape — e.g. when past *forecast errors*
carry information (MA terms), or when you need proper confidence intervals
around the forecast, which ARIMA provides and the smoothing methods above do
not.

---

## 12. Suggestions Specific to Wind Turbine SCADA, CMS, and High-Frequency Data

### SCADA data (typically 1-second to 10-minute averaged signals)

- SCADA systems usually log **10-minute averages** of wind speed, power output,
  pitch angle, rotor speed, and nacelle temperature. This averaging already
  smooths out a lot of turbulence — so **Holt-Winters with a daily (m=144 at
  10-min resolution) or yearly seasonal period** is a natural fit for capturing
  diurnal and seasonal wind patterns.
- Always plot **power curve** (wind speed vs. power output) first — SCADA data
  commonly has periods of curtailment (turbine deliberately throttled) or
  downtime that look like anomalies but are actually operational decisions,
  not sensor or forecasting problems.
- Missing data is common (communication dropouts) — before applying any of the
  methods above, decide on a consistent gap-filling strategy (e.g. linear
  interpolation for short gaps, leave longer gaps as missing rather than
  fabricating multi-hour trends).

### CMS data (Condition Monitoring System — vibration, bearing temperature, acoustic)

- CMS signals are usually sampled far more frequently (kHz range for vibration
  accelerometers) and are used for **fault/anomaly detection**, not just
  forecasting a value — the goal is often "does this look different from
  normal," not "what's the next value."
- Rather than forecasting raw high-frequency vibration directly, a common
  practical approach: extract **rolling statistical features** first (rolling
  mean, rolling standard deviation, RMS, kurtosis, dominant frequency via FFT)
  over short windows, *then* apply smoothing/forecasting or anomaly detection
  on those extracted features — this turns a noisy high-frequency signal into
  a much more forecastable, lower-frequency trend signal.
- SES/Holt on a rolling RMS or rolling standard deviation of vibration is a
  good simple baseline for **trending toward failure** (e.g. bearing wear
  showing as a slowly rising vibration RMS over weeks) — exactly the kind of
  slow upward trend Holt's method is built for.

### High-frequency data in general (sub-second to few-second sampling)

- Don't apply hand-calculation-style smoothing directly to raw high-frequency
  streams — first **downsample/aggregate** (e.g. 1-second data → 1-minute
  averages, or compute short-window features as above) so the noise level
  matches the timescale you actually care about forecasting.
- Turbulence at high frequency is often closer to random noise than genuine
  signal for forecasting purposes — a large part of the "art" here is picking
  an aggregation window that averages out turbulence while still preserving
  the trend/seasonal pattern you want to model (directly connects back to the
  moving-average window / SES α discussion in Sections 4–5).
- For genuinely high-frequency forecasting needs (e.g. very short-term power
  ramp prediction for grid balancing), simple methods like the ones in this
  document are typically used only as a **baseline** — production systems
  layer on NWP (numerical weather prediction) inputs, LSTM/Transformer models,
  or hybrid statistical+ML approaches, precisely because turbulence and
  multi-scale seasonality (daily + seasonal) exceed what ARIMA/Holt-Winters
  alone can capture well.

### General practical suggestion

Build up in this order when working with real turbine data: **(1)** plot the
raw signal and a rolling mean to see trend/seasonality visually, **(2)** try
naive/drift as a baseline, **(3)** try moving average and SES, **(4)** try
Holt's method if there's a clear trend, **(5)** try Holt-Winters if there's a
clear repeating cycle (daily/seasonal), **(6)** only then reach for ARIMA or
ML-based models — and always compare each new method's error (MAE/RMSE)
against the simple baseline to confirm the added complexity is actually
earning its keep.
