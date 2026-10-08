import numpy as np
import lightkurve as lk
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print('=' * 60)
print('EXOPLANET DETECTION — KEPLER-22b')
print('Using real NASA Kepler telescope data')
print('=' * 60)

# -------------------------------------------------------
# STEP 1: Download Data
# -------------------------------------------------------
print('\n[1/5] Searching for Kepler-22 light curve data...')
search_result = lk.search_lightcurve('Kepler-22', mission='Kepler', exptime=1800)
print(f'Found {len(search_result)} long cadence data products')

print('Downloading all quarters — this may take a minute...')
lc_collection = search_result.download_all()
lc_stitched = lc_collection.stitch()

# Flatten removes slow stellar variability, keeping only sharp transit dips
lc_flat = lc_stitched.flatten(window_length=1001)

# Remove outliers beyond 3 sigma
lc_clean = lc_flat.remove_outliers(sigma=3)
print(f'Total time span: {lc_clean.time.value[-1] - lc_clean.time.value[0]:.0f} days')
print(f'Total data points: {len(lc_clean)}')

# -------------------------------------------------------
# STEP 2: Plot Full Light Curve
# -------------------------------------------------------
print('\n[2/5] Plotting full light curve...')
fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(
    x=lc_clean.time.value,
    y=lc_clean.flux.value,
    mode='markers',
    marker=dict(size=1, color='black'),
    name='Flux'
))
fig_lc.update_layout(
    title='Kepler-22 Full Light Curve (1400+ days of real telescope data)',
    xaxis_title='Time (BKJD days)',
    yaxis_title='Normalised Flux',
    template='plotly_white',
    height=400
)
fig_lc.show()

# -------------------------------------------------------
# STEP 3: BLS Period Search
# -------------------------------------------------------
print('\n[3/5] Running Box Least Squares period search...')
print('Searching periods 200-350 days at 0.01 day resolution...')

# BLS searches for periodic box-shaped dips in brightness
# This is the standard algorithm used by professional astronomers
pg = lc_clean.to_periodogram(
    method='bls',
    period=np.arange(200, 350, 0.01)
)

best_period = pg.period_at_max_power
known_period = 289.86
error = abs(best_period.value - known_period)
error_pct = (error / known_period) * 100

print(f'Best period found:       {best_period:.2f}')
print(f'Known period of Kepler-22b: {known_period} days')
print(f'Error: {error:.2f} days ({error_pct:.1f}%)')

# Plot periodogram
fig_pg = go.Figure()
fig_pg.add_trace(go.Scatter(
    x=pg.period.value,
    y=pg.power.value,
    mode='lines',
    line=dict(color='navy', width=1),
    name='BLS Power'
))
fig_pg.add_vline(
    x=best_period.value,
    line_dash='dash',
    line_color='red',
    annotation_text=f'Detected: {best_period:.2f}'
)
fig_pg.add_vline(
    x=known_period,
    line_dash='dot',
    line_color='green',
    annotation_text=f'Known: {known_period} days'
)
fig_pg.update_layout(
    title='BLS Periodogram — Peak power indicates orbital period of Kepler-22b',
    xaxis_title='Period (days)',
    yaxis_title='BLS Power',
    template='plotly_white',
    height=400
)
fig_pg.show()

# -------------------------------------------------------
# STEP 4: Phase-folded Transit
# -------------------------------------------------------
print('\n[4/5] Folding light curve at detected period...')

# Folding at the detected period stacks all transits on top of each other to reveal the transit shape
lc_folded = lc_clean.fold(period=best_period)
# ~5 h bins: the transit lasts ~7 h, so this averages down noise without erasing the dip
lc_binned = lc_folded.bin(time_bin_size=0.2)

fig_fold = go.Figure()
fig_fold.add_trace(go.Scatter(
    x=lc_folded.time.value,
    y=lc_folded.flux.value,
    mode='markers',
    marker=dict(size=1, color='lightgray'),
    name='Raw folded flux',
    opacity=0.5
))
fig_fold.add_trace(go.Scatter(
    x=lc_binned.time.value,
    y=lc_binned.flux.value,
    mode='markers',
    marker=dict(size=5, color='navy'),
    name='Binned flux'
))
fig_fold.update_layout(
    title='Phase-folded Transit of Kepler-22b — Each point stacks multiple transits',
    xaxis_title='Phase (days)',
    yaxis_title='Normalised Flux',
    template='plotly_white',
    height=450,
    xaxis=dict(range=[-1, 1])
)
fig_fold.show()

# -------------------------------------------------------
# STEP 5: Planet Properties
# -------------------------------------------------------
print('\n[5/5] Estimating planet properties...')

transit_depth = 1 - lc_binned.flux.value.min()

# Transit depth = (Rp/Rs)^2
# Kepler-22 has radius 0.979 solar radii
star_radius_solar = 0.979
star_radius_earth = star_radius_solar * 109.076  # 1 solar radius = 109.076 Earth radii

planet_radius_earth = np.sqrt(transit_depth) * star_radius_earth

# Orbital distance from Kepler's third law: a = (P^2 * M)^(1/3)
# Kepler-22 has mass 0.97 solar masses
star_mass_solar = 0.97
# Use the detected period, so the distance is a real prediction rather than the known answer
period_years = best_period.value / 365.25
orbital_distance_au = (period_years**2 * star_mass_solar) ** (1/3)

print('\n' + '=' * 60)
print('RESULTS')
print('=' * 60)
print(f'Transit depth:           {transit_depth*100:.4f}%')
print(f'Planet radius:           {planet_radius_earth:.1f} Earth radii')
print(f'Orbital period:          {best_period.value:.2f} days')
print(f'Orbital distance:        {orbital_distance_au:.3f} AU')
print(f'(Earth is 1.0 AU from Sun)')
print(f'(Habitable zone:         ~0.8 - 1.8 AU for this star)')
print('=' * 60)
print(f'\nKnown values for comparison:')
print(f'True planet radius:      2.4 Earth radii')
print(f'True orbital distance:   0.849 AU')
print('=' * 60)
print('\nKepler-22b is in the habitable zone of its star.')
print('It was one of the first potentially habitable exoplanets discovered.')
print('Discovered by NASA Kepler mission, confirmed December 2011.')

# -------------------------------------------------------
# Summary Plot — all four panels together
# -------------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Full Light Curve',
        'BLS Periodogram',
        'Phase-folded Transit',
        'Planet Properties'
    )
)

# Full light curve
fig.add_trace(go.Scatter(
    x=lc_clean.time.value,
    y=lc_clean.flux.value,
    mode='markers',
    marker=dict(size=0.5, color='black'),
    name='Full LC',
    showlegend=False
), row=1, col=1)

# Periodogram
fig.add_trace(go.Scatter(
    x=pg.period.value,
    y=pg.power.value,
    mode='lines',
    line=dict(color='navy', width=1),
    name='BLS',
    showlegend=False
), row=1, col=2)

# Folded transit
fig.add_trace(go.Scatter(
    x=lc_folded.time.value,
    y=lc_folded.flux.value,
    mode='markers',
    marker=dict(size=1, color='lightgray'),
    name='Folded',
    showlegend=False,
    opacity=0.3
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=lc_binned.time.value,
    y=lc_binned.flux.value,
    mode='markers',
    marker=dict(size=4, color='navy'),
    name='Binned',
    showlegend=False
), row=2, col=1)

# Planet properties bar chart
properties = ['Planet Radius<br>(Earth radii)', 'Orbital Distance<br>(AU)', 'Period<br>(days / 10)']
detected = [planet_radius_earth, orbital_distance_au, best_period.value / 10]
known = [2.4, 0.849, known_period / 10]

fig.add_trace(go.Bar(
    name='Detected',
    x=properties,
    y=detected,
    marker_color='navy',
    showlegend=True
), row=2, col=2)
fig.add_trace(go.Bar(
    name='Known',
    x=properties,
    y=known,
    marker_color='lightcoral',
    showlegend=True
), row=2, col=2)

fig.update_layout(
    title='Kepler-22b Detection Summary — Using Real NASA Kepler Data',
    template='plotly_white',
    height=700,
    barmode='group'
)
fig.update_xaxes(range=[-1, 1], row=2, col=1)
fig.show()

print('\nDone. All plots generated.')