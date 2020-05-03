# On This Day 2 (OTD2)

While we can share OTD upon request, we also want to distribute a very similar dataset (OTD2) that
can be downloaded and used straight away.
For all intents and purposes, this dataset is an updated version of OTD.
The dataset, alongside its splits, can be found in the `data` folder as `otd2.pkl`.

## Data Collection
OTD2, like OTD, comprises a collection of events from the On This Day website, augmented with contextual information
obtained using the process that was outlined in our paper.
It was scraped in April 2020, and thus includes more recent events, alongside corrections made by the On This Day team
since then.
Notably, some events have been removed from the website since then, so the dataset is slightly smaller (71484 datapoints).
We preprocessed the dataset in the same way as we processed OTD, by removing events that happened before the year 1 CE and
that happened in the future (e.g., "31st predicted perihelion passage of Halley's Comet" in 2061 CE).
We provide 80/10/10 splits.

## Results
Below we also include the results of our two best performing model variants
(the bag-of-embeddings classifier, and the LSTM regressor).

*BoE model results coming soon...*


|                   | Kendall's Tau | Exact Match | Under 10Y | Under 50Y | Mean Absolute Error |
|-------------------|---------------|-------------|-----------|-----------|---------------------|
| LSTM (Reg, No CI) | 0.671±0.010   | 2.8±0.3     | 64.9±2.3  | 86.6±1.0  | 30.4±1.3            |
| LSTM (Reg, CI)    | 0.700±0.005   | 2.8±0.2     | 68.8±0.9  | 88.5±0.4  | 27.4±0.8            |