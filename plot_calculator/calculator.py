import code
import inspect
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import platform
import plot_calculator
import subprocess
import sympy as sp
import sympy
import webbrowser

from appdirs import user_cache_dir
from fractions import Fraction
from numpy import *
from numpy.linalg import *
from scipy.linalg import solve as solve2
# numpy.cos, sin.. replaced by explicit sympy imports
from sympy import Rational
from sympy import exp
from sympy import re, im, I, E
from sympy import pi
from sympy import sqrt
from sympy import cos, sin, tan, cot, sec, csc
# csc := cosecant
from sympy import acos, asin, atan, acot, asec
from sympy import sinh, cosh, tanh, coth, sech, csch
from sympy import asinh, acosh, atanh, acoth, asech, acsch
from sympy import deg, rad
from sympy import Eq, solve, solveset, symbols
from sympy import det, Matrix
from sympy import expand, pdiv, simplify


def suppress_qt_warnings():
    """Surpress matplotlib's QT dependencies warnings"""
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"


suppress_qt_warnings()

# Configure the size of the plots
plot_size = .35

# Get path of this module for the help text file location
module_dir = os.path.dirname(inspect.getfile(plot_calculator))


def open_help_text(filepath):
    # macos
    if platform.system() == 'Darwin':
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':
        os.startfile(filepath)
    else:
        # Linux
        subprocess.call(('less', filepath))


def help_calc():
    """Show calculation help text"""
    open_help_text(os.path.join(module_dir, "help-calc.txt"))


def help_plot():
    """Show plotting help text"""
    open_help_text(os.path.join(module_dir, "help-plot.txt"))


class MakePlot:
    """Matplotlib presettings - facilitates creating graphs."""

    def new_plot(self):
        """Create a new figure"""
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.tick_params(direction='out', length=plot_size*25, width=plot_size*8, colors='black')
        for axis in ['top','bottom','left','right']:
            self.ax.spines[axis].set_linewidth(plot_size*10)
        # Font Size of x- and y-axis tick labels
        plt.xticks(fontsize=plot_size*40)
        plt.yticks(fontsize=plot_size*40)

    def plot1(self, new_plot=True):
        """Plot the graph. See examples for how to create graph."""
        plt.legend(loc="upper right")
        leg = plt.legend()
        # Get the lines and texts inside legend box
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        # legend lines
        plt.setp(leg_lines, linewidth=plot_size*30)
        plt.setp(leg_texts, fontsize=plot_size*400)
        plt.legend(handletextpad=.8)
        # No grid preferred, because of overlayed grids in PDF editors
        # plt.grid(color='magenta', linestyle='dotted', linewidth=4)
        appname = "plot_calculator"
        appauthor = "plot_calculator"
        tmp_plot = os.path.join(user_cache_dir(appname, appauthor), "tmp-plot.png")
        pathlib.Path(user_cache_dir(appname, appauthor)).mkdir(parents=True, exist_ok=True)
        self.fig.savefig(tmp_plot)
        webbrowser.open(tmp_plot)
        # plt.show()
        if new_plot:
            self.new_plot()


    def plot2(self):
        """Plot inside the same figure"""
        plot1(new_plot=False)


def arctan(x):
    print("(W) For calculation better use sp.atan. For plotting: np.arctan")
    return np.arctan(x)


def arccos(x):
    print("(W) For calculation better use sp.acos. For plotting: np.arctan")
    return np.arccos(x)


def arcsin(x):
    print("(W) For calculation better use sp.asin. For plotting: np.arctan")
    return np.arcsin(x)


def deg(val):
    """Return degrees in decimals, not simplified pi form."""
    return sp.deg(val).n()


def clear():
    """Clear python CLI"""
    os.system('cls' if os.name == 'nt' else 'clear')


def frac1(fraction):
    """
    Use by default limit_denominator to simplify the fraction, else you get weird
    fractions.
    """
    return Fraction(fraction).limit_denominator()


def frac2(fraction):
    """Same as frac, but iterate over a list of decimals."""
    return [Fraction(x).limit_denominator() for x in fraction]


def solve(a, b):
    """Solve a linear equation system"""
    solutions = linalg.solve(a, b)
    print(solutions)
    return [frac(solution) for solution in solutions]


def round2(val):
    """Round decimal at the 6th decimal."""
    return around(val, decimals=6)


def start_calculator():
    """Script entry point"""
    pass


def R(frac):
    """Alias for sympy.Rational to enter fractions."""
    return Rational(frac)


# Set up plot initially
make_plot = MakePlot()
make_plot.new_plot()
ax = make_plot.ax
# Create a plot and set a new plot
plot1 = make_plot.plot1
# Plot inside existing plot
plot2 = make_plot.plot2

# Aliases
c = clear
help = help_calc
h = help_calc
hp = help_plot
lgs = solve
F = R
frac3 = R

# Default sympy symbols
a = symbols('a')
b = symbols('b')
x = symbols('x')
y = symbols('y')
z = symbols('z')
t = symbols('t')
u = symbols('u')
v = symbols('v')
w = symbols('w')

print("plot_calculator\n")
print("- help() - show all calculator commands\n")
print("- help_plot() to see examples for plotting graphs.\n")
code.interact(local=locals())
