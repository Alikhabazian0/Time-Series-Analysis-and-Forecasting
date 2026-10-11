# Time-Series Analysis and Forecasting

**Statistical modeling · Conditional volatility · Symbolic representations · Neural forecasting · Stochastic processes**

A collection of scientific computing notebooks investigating temporal dependence, volatility dynamics, symbolic representations, and forecasting in financial and synthetic time series. The repository combines ARIMA and GARCH-family models for Bitcoin data, symbolic discretization using SAX, and LSTM/Transformer experiments on a simulated stochastic process.

The notebooks provide an exploratory resource for computational physics, financial econometrics, and machine learning. They document methods and individual experiments; they do not constitute a standardized forecasting benchmark.

## Scientific objectives

- Investigate conditional mean and variance dynamics in Bitcoin log returns.
- Explore stationarity tests, correlation diagnostics, residual analysis, and information-criterion-based model comparison.
- Examine how symbolic representations change with discretization parameters.
- Compare neural one-step forecasts with simple baselines on a controlled drift–diffusion process.

## Repository contents

| Notebook | Main methods | Experiment |
| --- | --- | --- |
| [ARIMA model on BTC](bitcoin%20volatility/ARIMA%20model%20on%20BTC.ipynb) | ARIMA(1,0,1), ADF, ACF/PACF, Ljung–Box | Model Bitcoin log returns, examine residuals, and construct an illustrative price projection. |
| [ARIMA-GARCH on BTC](bitcoin%20volatility/ARIMA-GARCH%20on%20BTC.ipynb) | Automatic ARIMA selection, GARCH(1,1) | Fit conditional variance to ARIMA residuals and inspect volatility dynamics. |
| [GARCH on BTC](bitcoin%20volatility/GARCH%20on%20BTC.ipynb) | GARCH, EGARCH, GJR-GARCH, AIC/BIC | Compare volatility specifications, forecast volatility, simulate price scenarios, and explore tail-return statistics. |
| [SAX on BTC](symbolic%20representations/SAX%20on%20BTC.ipynb) | Symbolic Aggregate Approximation, quantile discretization | Explore symbolic encodings, alphabet sizes, and rolling-window experiments on Bitcoin closing prices. |
| [Trend Following: Transformers and LSTM](synthetic%20forecasting/Trend%20Following%20-%20Masked%20Transformers%20and%20LSTM.ipynb) | TensorFlow/Keras LSTM, PyTorch Transformer encoder | Compare one-step forecasts of a simulated process, with linear-regression and drift baselines. |

The [Results directory](Results/) contains saved figures. The Bitcoin CSV is stored in [symbolic representations](symbolic%20representations/BTCUSD-2y.csv).

## Methodological framework

### Returns and conditional mean modeling

Bitcoin closing prices $P_t$ are transformed into log returns:

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
$$

ARIMA models describe linear dependence in the conditional mean. For the ARIMA(1,0,1) example, the model can be written as:

$$
r_t = c + \phi r_{t-1} + \varepsilon_t + \theta\varepsilon_{t-1}.
$$

The notebooks use the augmented Dickey–Fuller test, autocorrelation and partial autocorrelation plots, residual distributions, and Ljung–Box diagnostics. The combined ARIMA–GARCH notebook selects an ARIMA specification using `pmdarima.auto_arima` and then models its residual variance.

### Conditional volatility modeling

For a GARCH(1,1) process:

$$
\varepsilon_t = \sigma_t z_t,
\qquad
\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2,
$$

where $\sigma_t^2$ is the conditional variance and $z_t$ is a standardized innovation. The volatility notebook compares GARCH(1,1), GARCH(2,1), EGARCH(1,1), and GJR-GARCH(1,1) using AIC and BIC. These comparisons concern fitted models; the subsequent forecasting cells use the separately fitted GARCH(1,1) object.

The notebook also constructs seven-step price scenarios from zero-mean Gaussian returns scaled by forecast volatility. It reports lower-tail quantiles and tail means of cumulative simulated log returns, alongside a historical-quantile exercise. These are illustrative calculations rather than an independently validated risk forecast.

### Symbolic representations

The SAX notebook uses `pyts.approximation.SymbolicAggregateApproximation` to map numerical observations to a finite alphabet. It explores different alphabet sizes and quantile-based discretization, visualizes the resulting symbols, and includes rolling-window experiments.

The current examples focus on symbolic discretization. They do not establish a complete, validated SAX compression or forecasting pipeline; preprocessing, aggregation, and downstream evaluation should be specified for a particular research task.

### Neural forecasting of a stochastic process

The synthetic example generates a drift–diffusion process:

$$
dX_t = \mu\,dt + g\,dW_t,
$$

using the discrete update:

$$
X_{n+1} = X_n + \mu\Delta t + g\sqrt{\Delta t}\,\xi_n,
\qquad \xi_n \sim \mathcal{N}(0,1).
$$

The notebook uses 25-observation input windows and a chronological 80/20 split for one-step prediction. It implements:

- A TensorFlow/Keras LSTM with 50 units and a scalar output.
- A PyTorch Transformer encoder with 64-dimensional embeddings, four attention heads, and three encoder layers.
- A linear trend extrapolation baseline and a one-step baseline using drift estimated from training increments.

Prediction errors are reported as mean squared error (MSE) and root mean squared error (RMSE). For this constant-drift process, the conditional mean of the next observation is $X_n + \mu\Delta t$; tracking the level of a random walk should therefore be assessed against a persistence/drift baseline before being interpreted as additional predictive skill.

## Data

The included `BTCUSD-2y.csv` contains **725 observations** spanning **2 April 2023 to 1 April 2025**, with columns:

```text
Date, Timestamp, Open, High, Low, Close, Volume
```

`Date` is encoded as `YYYYMMDD`. Inspect missing dates and sampling intervals before assuming equally spaced observations. The repository does not document the original data provider or volume units.

The synthetic data are generated within the forecasting notebook; no separate download is required.

## Installation and execution

Clone the repository:

```bash
git clone https://github.com/Alikhabazian0/Time-Series-Analysis-and-Forecasting.git
cd Time-Series-Analysis-and-Forecasting
```

In a dedicated Python environment, install the packages used by the notebooks:

```bash
python -m pip install numpy pandas scipy matplotlib seaborn statsmodels pmdarima arch pyts scikit-learn tensorflow torch jupyterlab ipykernel
python -m jupyterlab
```

Select the kernel associated with that environment. Dependencies are not version-pinned, so this command is a starting point rather than a verified reproduction environment.

### Update data paths

The Bitcoin notebooks currently load the CSV through absolute Windows paths. Replace those paths with the location of the included file. The following example supports a working directory at the repository root or inside one of its notebook folders:

```python
from pathlib import Path
import pandas as pd

repo_root = Path.cwd()
data_path = repo_root / "symbolic representations" / "BTCUSD-2y.csv"
if not data_path.exists():
    repo_root = repo_root.parent
    data_path = repo_root / "symbolic representations" / "BTCUSD-2y.csv"

df = pd.read_csv(data_path, dtype={"Date": str})
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
df = df.sort_values("Date")
```

For the statistical notebooks, also use `df = df.set_index("Date")`. Replace their existing loading cells rather than running both date-parsing approaches. Start with ARIMA, continue to GARCH and ARIMA–GARCH, then explore SAX and the synthetic forecasting experiment.

## Example outputs

**Volatility-based Bitcoin price scenarios**

![Saved GARCH-based Bitcoin price scenarios](Results/GARCH-Based%20Simulated%20BTC%20Price%20Paths%20for%207%20Days.png)

**LSTM and Transformer forecasts on a synthetic process**

![Saved LSTM and Transformer predictions](Results/LSTM%20and%20Masked%20Transformer%20on%20RW.png)

Figures reflect individual saved runs. Random initialization, sampling, and environment changes can alter the outputs.

## Interpretation and reproducibility

Several details matter when extending these exploratory notebooks into a quantitative study:

- **Transformer specification:** the class is named `MaskedTransformer`, but its current forward pass supplies no attention mask or positional encoding. Its inputs contain historical windows; the implementation should be described as an encoder rather than a demonstrated causally masked architecture.
- **Forecast evaluation:** the neural examples fit scalers before splitting the series and use the held-out LSTM segment for validation. Fit preprocessing on training data only, introduce a separate validation period, and compare all models on identical targets and horizons.
- **Symbolic model selection:** the SAX parameter loop scores a classifier on its training observations, and its `window_size` loop variable does not change the transformation in that loop. The reported choice is not evidence of out-of-sample performance.
- **Uncertainty and risk:** cumulative sums of marginal ARIMA interval endpoints are not calibrated cumulative-price prediction intervals. The price simulation uses forecast volatility without recursively updating variance for each simulated path. Its seven-step tail statistics and one-step historical quantiles have different horizons; the historical violation count is not an out-of-sample GARCH VaR backtest.
- **Diagnostics and reporting:** inspect actual test outputs rather than treating notebook commentary as universal conclusions. The GARCH notebook's ADF p-value print statement repeats the test statistic, and the ARIMA–GARCH Ljung–Box unpacking should be updated for the installed `statsmodels` API.
- **Repeatability:** record dataset provenance, preprocessing, package versions, random seeds, and training settings. Use repeated stochastic realizations and rolling or expanding evaluation windows before drawing general conclusions.

These considerations define the limits of the present experiments and provide a starting point for more rigorous comparisons.

## References

1. Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity*. Journal of Econometrics, 31(3), 307–327. [DOI](https://doi.org/10.1016/0304-4076(86)90063-1).
2. Lin, J., Keogh, E., Lonardi, S., and Chiu, B. (2003). *A symbolic representation of time series, with implications for streaming algorithms*. Proceedings of the 8th ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge Discovery. [Authors' SAX resource](https://www.cs.ucr.edu/~eamonn/SAX.htm).
3. Hochreiter, S., and Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation, 9(8), 1735–1780. [DOI](https://doi.org/10.1162/neco.1997.9.8.1735).
4. Vaswani, A., et al. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems, 30. [Paper](https://arxiv.org/abs/1706.03762).

These references describe methodological foundations; the notebooks are exploratory implementations of selected ideas.

## Maintainer

[Ali Khabazian](https://github.com/Alikhabazian0)
