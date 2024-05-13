# %%

from pypenalty import penalty, smooth_square_root, power
import numpy as np
import proplot as pplt

lb = -1
ub = 1
x = np.linspace(-2, 2, 500)

y_sqr = penalty(x, lb, ub, epsilon=0.1, weight=1.0, fct=smooth_square_root)
y_pow = penalty(x, lb, ub, power=2, weight=1.0, fct=power)

fig, ax = pplt.subplots(figsize=(5, 4))
ax.plot(x, y_sqr, label="Smooth square root")
ax.plot(x, y_pow, label="Power")

ax.vlines(lb, -0.05, 1.0, color="k", linestyle="--")
ax.vlines(ub, -0.05, 1.0, color="k", linestyle="--")

ax.format(
    xlabel="x",
    ylabel="Penalty value",
    # ylim=(-0.0001, 0.0001),
)
ax.legend(ncols=1)
pplt.show()
# fig.savefig("docs/img/penalty.png")

# %%
