# Deep Momentum Networks for Time Series Momentum Strategies

## Overview

This project repository implements the deep momentum network timeseries momentum factor proposed by [Lim, Zohren and Roberts (2019)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3369195).

For a description of the full results as well as a summary of time series momentum strategies, please confer the [pdf report](https://github.com/maxlamberti/time-series-momentum/blob/master/Report.pdf). 

The plot below summarizes the performance of the TSMOM-DMN factor (with and without transaction cost) against the AQR cross-sectional momentum factor and a baseline time series momentum strategy (SIGN).

<img src="https://github.com/maxlamberti/time-series-momentum/blob/master/plots/comparison_price_log_series.png" width="550">

## Data

The data originates from the Pinnacle Data Corp [CLC database](https://pinnacledata2.com/clc.html) of continuously linked futures contracts.

## Context

This report was created as part of the Dynamic Asset Management course at the Berkeley MFE.
