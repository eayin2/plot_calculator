"""
Ex.: Möbius Transformation of a grid with 8 lines ranging from (-2, 2)

Paste the code into the interactive calculator shell
"""

# Set a high range, as the grid lines should be "infinite long" and considered a circle
# with an infinite radius for the Möbius transformation
t = np.arange(-2000, 2000, 0.08)
for i in range(-2,3):
    # Horizontal grid line
    gh = t + i*1j
    fh = (1j*gh-2)*((1+1j)*gh-(2+1j))**-1
    ax.plot(fh.real, fh.imag, label="f_h%s(z)" % x, linewidth=plot_size*10)
    # Vertical grid line
    gv = i + t*1j
    fv = (1j*gv-2)*((1+1j)*gv-(2+1j))**-1
    ax.plot(fv.real, fv.imag, label="f_v%s(z)" % x, linewidth=plot_size*10)

plot()
