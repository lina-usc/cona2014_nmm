# Cona et al (2014) Neural Mass Model of the Thalamocortical Loop

### Usage

The code

```python
import matplotlib.pyplot as plt
from cona2014_nmm import get_all_param, thal_ctx_simulator

allpar = get_all_param()
np_data = thal_ctx_simulator(allpar)

def min_max_norm(x):
    return (x - x.min())/(x.max() - x.min())

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

t = allpar["t"]
axes[0].plot(t, min_max_norm(allpar["modulTCN"]), "-", label='TC')
axes[0].plot(t, min_max_norm(allpar["modulTRN"]), "--", label='Reticular')
axes[0].plot(t, min_max_norm(allpar["modulP"]), "-.", label='Pyramidal')

axes[0].set_title('From wakefulness to deep sleep')
axes[0].set_ylabel('Modulatory inputs\n(normalized)')
axes[0].legend(loc="right")#title="Input")


axes[1].plot(t[t >=30][::10], np_data["vp"].squeeze())
axes[1].set_xlim(30, 105)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Membrane\npotential (mV)')

for ax in axes:
    for x in range(45, 91, 15):
        ax.axvline(x=x, color="k", linestyle="dashed", alpha=0.2)  
```

reproduces the two first panels of Figure 11 of Cona et al (2014). The discrepancy in the duration of the noisy transition around t=80s is due to a discrepancy in the paper between signals shown in the first panel and the signals used to produce the second panel. In the signals used by the authors, the drop in the TC modulatory signal is much faster, shortening that transitory stage.

