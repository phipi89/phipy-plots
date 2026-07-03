# phipy-plots

A collection of plotting utilities and styling presets I keep reusing across projects.

Notable tools:
 - `pp.Figure`: Convenience wrapper around plotly as I never got warm with plotly express.
 - `Colors` / `colors` / `red|green|blue|…`: Color helpers and a reusable color collection that works with both plotly and matplotlib.

```python
import phipy_plots as pp

fig = Figure()

fig._mesh(sphere, color=blue.lighter())
fig._curve(spiral, color=red)

fig._view(azim=10, elev=40)
fig.show()
```
