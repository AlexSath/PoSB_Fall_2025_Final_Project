from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

def visualize_single_var_axis(
        key: str,
        amplitudes: NDArray,
        delays: NDArray,
        axis_titles: bool = False
):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.scatter(np.arange(len(amplitudes)), amplitudes)
    ax1.set_xticks(np.arange(len(amplitudes)))
    ax1.set_xticklabels(
        [f"1/{(int(2*x))}x" for x in np.arange(1, len(amplitudes) / 2)][::-1] \
        + ["1x"] \
        + [f"{int(2*x)}x" for x in np.arange(1, len(amplitudes) / 2)]
    )
    ax1.set_xlabel("Parameter Perturbation")
    ax1.set_ylabel("Peak Output Amplitude (REU)")
    ax1.set_ylim(0, 250)
    if axis_titles: ax1.set_title("Amplitude")

    ax2.scatter(np.arange(len(delays)), delays)
    ax2.set_xticks(np.arange(len(delays)))
    ax2.set_xticklabels(
        [f"1/{(int(2*x))}x" for x in np.arange(1, len(delays) / 2)][::-1] \
        + ["1x"] \
        + [f"{int(2*x)}x" for x in np.arange(1, len(delays) / 2)]
    )
    ax2.set_xlabel("Parameter Perturbation")
    ax2.set_ylabel("Peak Output Delay (s)")
    ax2.set_ylim(0, 5500)
    if axis_titles: ax2.set_title("Output Delay")
    fig.suptitle(key)
    plt.tight_layout()

def visualize_two_var_grid(
        data: NDArray,
        keys: list[str],
        plt_title: str,
        cbar_min: float=None,
        cbar_max: float=None,
        cbar_step: float=None,
        cbar_title: str=None,
        divide_cbar_label_by: float=None
):
    im = plt.imshow(data)
    plt.ylabel(keys[0])
    plt.xlabel(keys[1])
    plt.title(plt_title)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(
        [f"1/{(int(2*x))}x" for x in np.arange(1, len(data) / 2)][::-1] \
        + ["1x"] \
        + [f"{int(2*x)}x" for x in np.arange(1, len(data) / 2)],
        rotation = 45
    )
    ax.set_yticklabels(
        [f"1/{(int(2*x))}x" for x in np.arange(1, len(data) / 2)][::-1] \
        + ["1x"] \
        + [f"{int(2*x)}x" for x in np.arange(1, len(data) / 2)],
    )
    ax.tick_params(
        axis='both',
        which='both',
        length=0
    )
    cbar = plt.colorbar()
    if cbar_step is None: cbar_step = 1 # default step is 1
    if cbar_min is None: cbar_min = np.floor(cbar.get_ticks()[0] / cbar_step) * cbar_step
    if cbar_max is None: cbar_max = np.ceil(cbar.get_ticks()[-1] / cbar_step) * cbar_step
    im.set_clim(cbar_min, cbar_max)
    new_ticks = np.arange(cbar_min, cbar_max + cbar_step, cbar_step)
    cbar.set_ticks(new_ticks)
    if divide_cbar_label_by is not None: cbar.set_ticklabels((new_ticks / divide_cbar_label_by))
    else: cbar.set_ticklabels(new_ticks.astype(int))
    if cbar_title is not None: cbar.set_label(cbar_title)
    plt.grid(visible=False)
    plt.tight_layout()