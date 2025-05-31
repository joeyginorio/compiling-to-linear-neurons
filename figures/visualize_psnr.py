
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['axes.unicode_minus'] = False      # use '-' instead of '−'


df5 = pd.read_csv("data/direct_psnr_test_final.csv")
idf5 = pd.read_csv("data/indirect_psnr_test_final.csv")

df5["model"]  = "D"     # name them as you like
idf5["model"] = "I"

bmidnight = (0, .329, .576)
bcayenne = (.8, 0, .14)

palette = {
    "D":   bmidnight,      # bmidnight
    "I": bcayenne       # bcayenne
}
hue_order = ["D", "I"]

sns.set_theme(style="white",     
              font="Futura",
              rc={                                      # global overrides
                "font.weight": "bold",
                "xtick.labelsize": 15,   # ← increase these numbers
                "ytick.labelsize": 15})

# ── 2. concatenate ───────────────────────────────────
big = pd.concat([df5, idf5], ignore_index=True)

g = sns.relplot(
    data=big,
    kind="line",
    x="step", y="psnr",
    col="batch size",           # share x-axis within each batch-size column
    row="lr",
    hue="model",
    hue_order=hue_order,
    palette=palette, 
    estimator="mean",           # average over seeds
    ci="sd",                    # ±1 SD ribbon
    facet_kws=dict(sharex="col", sharey=True),
    height=2.8, aspect=1.2
)

g.set_axis_labels("step", "PSNR")


# ── 1️⃣  pull out a master copy of the handles/labels ─────────────
first_ax = g.axes[0, 0]                              # upper-left panel
handles_all, labels_all = first_ax.get_legend_handles_labels()

# ── 2️⃣  drop Seaborn’s single, grid-level legend ────────────────
if g._legend is not None:
    g._legend.remove()

# ── 3️⃣  optional: blank the facet titles ─────────────────────────
g.set_titles("")          # comment out if you still want them

# ── 4️⃣  stitch a custom legend into *every* subplot ─────────────
for r, lr_val in enumerate(g.row_names):             # rows = learning rates
    for c, bs_val in enumerate(g.col_names):         # cols = batch sizes
        ax = g.axes[r, c]

        # -- build context-rich labels like "Model-A  (bs=32, lr=1e-3)"
        local_labels = [
            f"{lab}  (bs:{bs_val}, lr:{lr_val})"
            for lab in labels_all
        ]

        ax.legend(
            handles_all, local_labels,
            loc="lower right",     # place in bottom-right corner
            fontsize=13.5,
            frameon=True,
            title=None
        )

g.set_axis_labels("", "")        # wipes the “step” / “PSNR” strings
#   • supxlabel / supylabel are available in Matplotlib ≥ 3.4
g.fig.supxlabel("Steps",  y=-0.01, fontsize=24, fontweight="bold")      # global x-axis label
g.fig.supylabel("PSNR", x=-0.01, fontsize=24, fontweight="bold")       # global y-axis label
g.fig.suptitle("PSNR Dynamics", y=1.02, fontsize=28, fontweight="bold")       # global y-axis label

g.fig.savefig(
    "figures/psnr_global.pdf",
    format="pdf",
    dpi=900,              # controls resolution of any raster elements
    bbox_inches="tight",  # trims excess whitespace
    pad_inches=0.02       # small padding around the figure
)

plt.show()