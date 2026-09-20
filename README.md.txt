# QQQ Quantitative Market Regime Research

A Python-based quantitative research dashboard designed to analyze **QQQ / Nasdaq-100 market regimes** using options data, volatility, positioning, statistical methods, and cross-asset relationships.

The project combines derivatives mathematics with statistical analysis and automated visualization to create a daily research snapshot of the market.

> **Important:** This is a research and analytics project, not a validated trading strategy or a claim of predictive profitability.

---

## What This Project Does

The dashboard combines several areas of quantitative finance:

* Options Greeks and dealer-positioning proxies
* Gamma exposure (GEX)
* Vanna exposure (VEX)
* Charm exposure (CHEX)
* Implied volatility and volatility skew
* 0DTE options analysis
* Realized volatility regimes
* Hurst exponent estimation
* CFTC futures positioning
* Cross-asset correlation
* Lead/lag correlation
* Seasonality
* Statistical anomaly detection
* Historical event analysis
* Intraday VWAP and price structure
* Automated research summaries and visualizations

The goal is to study how these variables interact rather than rely on a single indicator.

---

## Quantitative Methods

### Black-Scholes Option Greeks

The project calculates option Greeks directly from option-chain data using the Black-Scholes framework.

For each option:

$$
d_1 =
\frac{\ln(S/K)+(r+\frac{1}{2}\sigma^2)T}
{\sigma\sqrt{T}}
$$

where:

* \(S\) = underlying price
* \(K\) = strike price
* \(T\) = time to expiration
* \(r\) = assumed risk-free rate
* \(\sigma\) = implied volatility

Gamma is calculated as:

$$
\Gamma =
\frac{\phi(d_1)}
{S\sigma\sqrt{T}}
$$

The project also calculates Vanna and Charm from the same Black-Scholes framework.

### Gamma Exposure (GEX)

Gamma is converted into an exposure estimate using option open interest.

Conceptually:

$$
GEX \propto
\Gamma \times OI \times 100 \times S^2 \times 0.01
$$

Calls and puts are assigned opposite signs under the project's positioning assumption.

The dashboard then calculates:

* Net GEX
* Gamma by strike
* Cumulative GEX
* Gamma Flip
* Call Wall
* Put Wall
* GEX by expiration

The Gamma Flip is the strike where cumulative gamma exposure crosses zero.

### Vanna Exposure

Vanna measures how option delta changes as implied volatility changes.

The dashboard expresses VEX as an estimated dollar delta change for a 1-volatility-point move.

### Charm Exposure

Charm measures how delta changes as time passes.

The project converts Charm into an estimated daily dollar delta change.

---

## Volatility Analysis

The project measures several forms of volatility:

### Realized Volatility

20-day and 60-day realized volatility are calculated from logarithmic returns:

$$
HV =
\sigma_{returns}\sqrt{252}
$$

This is compared with the VIX to distinguish recent realized volatility from implied volatility.

### Implied Volatility

The dashboard calculates:

* ATM implied volatility
* 5% OTM put volatility
* 5% OTM call volatility
* Put/call volatility skew
* Volatility risk premium

It also creates a small volatility surface across multiple strikes and expirations.

---

## Hurst Exponent

The project estimates the Hurst exponent using **multiple R/S windows followed by log-log linear regression**.

The general relationship is:

$$
E[R/S] \propto n^H
$$

where \(H\) is estimated from the slope of:

$$
\log(R/S)
\quad \text{vs.} \quad
\log(n)
$$

The implementation uses multiple window sizes rather than relying on a single-window estimate.

This is used as a research measure of persistence versus mean-reverting behavior.

---

## Positioning Data

The dashboard retrieves Nasdaq-100 futures positioning from the CFTC's public data API.

It calculates:

$$
Net\ Position =
Long\ Positions - Short\ Positions
$$

and normalizes this by total open interest.

This provides a weekly positioning measure that can be compared with options and volatility conditions.

---

## Statistical Research

Additional statistical features include:

### Seasonality

Historical QQQ returns are grouped by calendar month to calculate:

* Average monthly return
* Historical win rate
* Number of observations

### Correlation

Rolling return relationships are calculated between QQQ and:

* SPY
* DXY
* BTC

### Lead/Lag Analysis

The project compares same-day relationships with lagged relationships to investigate whether previous-day moves in BTC or DXY are associated with the following day's QQQ return.

### Anomaly Detection

Historical dashboard observations are stored and used to calculate z-scores for variables such as:

* Net GEX
* VIX
* ATM IV
* CFTC positioning

The system waits for sufficient historical observations before reporting an anomaly.

### Gamma Velocity

Gamma velocity measures the recent rate of change of Net GEX to identify whether the estimated gamma environment is moving toward more dampening or more amplifying conditions.

---

## 0DTE and Options Positioning

For options expiring the same day, the dashboard calculates:

* 0DTE Net GEX
* Max Pain
* Put/Call Open Interest Ratio
* Remaining time to expiration

Max Pain is calculated by evaluating the total option payoff across possible settlement strikes and identifying the strike with the lowest aggregate payout.

---

## Intraday Analysis

The dashboard also includes:

* Intraday QQQ price
* Session VWAP
* Previous-day high/low
* ATR-based expected range
* Gamma Flip
* Call Wall
* Put Wall
* Fair Value Gap detection
* Order-block heuristics
* Overnight range

The FVG and order-block components are heuristic research features rather than standalone trading signals.

---

## Composite Confluence Score

The dashboard contains a hand-weighted composite score ranging from approximately **-100 to +100**.

It combines:

* CFTC positioning
* Implied-volatility skew
* Historical seasonality

The score is explicitly a **heuristic**, not a trained machine-learning model or statistically validated alpha model.

The dashboard also includes a "Devil's Advocate" section that checks whether individual components disagree with the overall composite reading.

---

## Data Sources

The project currently uses publicly available data through:

* **Yahoo Finance / yfinance**

  * QQQ
  * QQQ options chains
  * VIX
  * SPY
  * DXY
  * BTC
  * NQ futures
* **CFTC public API**

  * Nasdaq-100 futures positioning
* **Chart.js**

  * Dashboard visualization

The program uses fallback/error handling so that a failure in one data source does not necessarily prevent the rest of the dashboard from being generated.

---

## Important Assumptions & Limitations

### QQQ vs. NQ

The primary options analysis uses **QQQ as a proxy for Nasdaq-100 / NQ analysis**.

QQQ and NQ are highly related but are not identical instruments, particularly during overnight trading.

The Pine Script export therefore scales QQQ-derived levels to the current NQ/QQQ price ratio rather than calculating those levels from NQ options directly.

### Dealer Positioning

The sign convention assumes a simplified dealer-positioning model:

* Calls contribute positively
* Puts contribute negatively

This is a commonly used approximation, but it is **not direct access to actual dealer books**.

### Black-Scholes Assumptions

The Greek calculations rely on assumptions inherent in the Black-Scholes framework, including the use of a constant risk-free-rate input and the observed implied volatility from the available option chain.

### Historical Relationships

Correlation, seasonality, positioning, and volatility relationships can change over time.

A relationship visible in historical data should not automatically be interpreted as a persistent trading edge.

---

## Engineering Features

The project is also designed as a small quantitative research pipeline.

It includes:

* Automated market-data retrieval
* Data-processing and calculation layers
* Error logging
* Historical state storage
* Daily dashboard generation
* Archive snapshots
* Interactive HTML charts
* Pine Script level generation
* Graceful handling of unavailable data

The main output is an automatically generated HTML research dashboard.

---

## Current Research Status

This project should currently be viewed as a **quantitative research and market-regime analysis platform**.

The major next step is systematic validation.

Future research can test questions such as:

1. Does negative estimated GEX correspond to higher future realized volatility?
2. Does extreme options positioning correspond to different future return distributions?
3. Does implied volatility contain information about future realized volatility beyond recent realized volatility?
4. Do combinations of these variables provide information out of sample?

Those questions require proper time-series backtesting, transaction costs, slippage assumptions, statistical testing, and out-of-sample validation.

---

## Disclaimer

This project is for educational and research purposes only.

Nothing in this repository constitutes investment advice, a recommendation to trade, or evidence of a profitable trading strategy.
