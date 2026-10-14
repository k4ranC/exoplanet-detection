# Exoplanet Detection: Kepler-22b

Detects the orbital period of a real exoplanet from raw NASA Kepler telescope data, using the transit method: a planet crossing its star makes the star dim slightly, once per orbit.

## What it does

1. Downloads every long-cadence Kepler light-curve quarter for Kepler-22 and stitches them together.
2. Flattens slow stellar variability and clips 3-sigma outliers.
3. Runs a Box Least Squares (BLS) period search over 200-350 days.
4. Folds the light curve at the best period to stack the transits.
5. Estimates planet radius from transit depth and orbital distance from Kepler's third law.

The detected period is compared against the published value (289.86 days) and the error is printed, so the result is checked rather than assumed.

## Run it

```
pip install -r requirements.txt
python exoplanet_detection.py
```

The first run downloads the data from MAST, which takes a minute or two. Plots open in the browser.

## Stack

Python, NumPy, Lightkurve, Plotly.
