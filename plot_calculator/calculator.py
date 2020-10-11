import io
import code
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
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


class MakePlot:
    """Matplotlib presettings"""

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


# Calculator functions


def arctan(x):
    print("(W) Better use atan (sympy)")
    return np.arctan(x)


def arccos(x):
    print("(W) Better use acos (sympy)")
    return np.arccos(x)


def arcsin(x):
    print("(W) Better use asin (sympy)")
    return np.arcsin(x)


def deg(val):
    return sp.deg(val).n()


def clear():
    """Clear python CLI"""
    os.system('cls' if os.name == 'nt' else 'clear')


def frac(fraction):
    """Use limit_denominator to simplify the fraction, else you get weird fractions"""
    return Fraction(fraction).limit_denominator()


def frac2(fraction):
    """Same as frac, but iterates over a list of values to simplify each to a fraction.
    Use limit_denominator to simplify the fraction, else you get weird fractions"""
    return [Fraction(x).limit_denominator() for x in fraction]


def solve(a, b):
    """Solve a linear equation system"""
    solutions = linalg.solve(a, b)
    print(solutions)
    return [frac(solution) for solution in solutions]


def round2(val):
    return around(val, decimals=6)


lgs = solve


def help():
    print(""
        "Calculator usage:\n"
        "- help() and clear()\n"
        "- 2**2 - 2 to the power of 2. Don't use 2^2\n"
        "- atan(-2/2) will return the decimal value. Use atan(-1) instead to get -pi/4 \n"
        "Fraction with sympy\n"
        "- Rational('1/3') - E.g.: 3*Rational('1/3')*Rational('2/5')\n"
        "\n"
        "Complex numbers with sympy\n"
        " a = x+y*I\n"
        " a.expand()\n"
        "\n"
        "Complex numbers with numpy using `1j`\n"
        " a = 2+3*1j\n"
        "\n"
        "Custom function:\n"
        "- frac(3.75) - A decimal to a fraction\n"
        "- frac2([3.75, 2.5, 0.5]) - List of decimal to fractions\n"
        "- Use imported modules: numpy, scipy, sympy. E.g. scipy.solve()\n"
        "\n"
        "Polynoms - sympy\n"
        "- pdiv(x**2-x, x-1) - Polynom division. Requires one root. E.g. x-1 for root x=1\n"
        "- expand((2*x-1)*(x**2-4)) - Expand term\n"
        "- simplify(2*x**3+4*x**3+2*x+3*x) - Simplify term\n"
        "\n"
        "Equation equal sympy:\n"
        "- exp(I*3/2*pi) - exp(-I*1/2*pi) == 0\n"
        "- or also: exp(I*3/2*pi).equals(exp(-I*1/2*pi))\n"
        "\n"
        "Solve equation sympy:\n"
        "- solve(a, b) or lgs(a, b) - Solve linear equation system. E.g.:\n"
        "    a = array([[1,2], [0, 3]])\n"
        "    b = array([5, 4])\n"
        "    solve(a, b)\n"
        "  Be aware: [1, 2, 4]**T transposes to a column vector and has to be added to the"
        " matrix as a column.\n"
        "- solveset(Eq(x, 1), x) Solve equation: \n"
        "  x = symbols('x')\n"
        "  solveset(Eq(x**2+x, 2), x)\n"
        "\n"
        "Trigonometric numpy:\n"
        "- degrees() - numpy.degrees - Radiant to degrees\n"
        "- radians() - numpy.radians - Degrees to radiant. E.g. sin(radians(90))\n"
        "- arccos() - Arcuscosinus\n"
        "\n"
        "Trigonometric sympy:\n"
        "- deg() - Sympy.deg - Radiant to degrees. Patched to return absolute value by default\n"
        "- rad() - Sympy.rad - Radiant to degrees\n"
        "- <object>.n() - (Sympy) Show not simplified result, but absolute result with"
        " decimals. E.g. cos(rad(40)).n()\n"
        "- More.. cos, sin, tan, cot, acos, asin, atan, acot, sinh, cosh, tanh, coth, asinh..\n"
        "\n"
        "Linear Algebra scipy:\n"
        "- solve2() - scipy.solve - Solve matrix equation such as `A*X = B`\n"
        "  E.g.: solve2(array([[3,5],[10,17]]), array([[1,2],[0,3]]))\n"
        "\n"
        "Linear Algebra sympy:\n"
        "- Matrix() - sympy.matrix. E.g. sympy.dot requires sympy.matrix and not"
        " numpy.array()\n"
        "- Matrix(array(..)) - Convert numpy.array to Matrix\n"
        "- det() - Determinant\n"
        "  Example: det(Matrix([[1,2,5],[3,-4,7],[-3,12,-15]]))\n"
        "  sympy.det\n"
        "\n"
        "Linear Algebra numpy:\n"
        "- transpose() - Transposed matrix. E.g. transpose(array([1,2,3]))\n"
        "- cross() - Cross product\n"
        "- dot() - Scalar or dot product\n"
        "- inv() - A**(-1), Inverse of a matrix\n"
        "\n\n"
    )


def help_plot():
    print(
        'Example plots\n'
        '\n'
        'Ex. 1: Trigonometric functions with pi\n'
        'x = np.arange(-2*np.pi, 2*np.pi, 0.001)\n'
        'ax.plot(x, np.tan(x), label="tan(x)", linewidth=plot_size*10)\n'
        'ax.plot(x, np.cos(x), label="cos(x)", linewidth=plot_size*10)\n'
        '# Set legend\n'
        'ax.legend(loc="upper right")\n'
        '# Set axis\n'
        'ax.axis([-2*np.pi, 2*np.pi, -2, 2])\n'
        'plot1\n'
        '\n'
    )
    print(
        'Ex. 2: Formatted labels for graph:\n'
        'ax.plot(x, np.sqrt(x**2-1), label=r"$\frac{e^x}{2}$", linewidth=plot_size*0.5)\n'
        'ax.plot(np.cosh(x), np.sinh(x), label=r"$coshx, sinhx$", linewidth=plot_size*0.5)\n'
        '\n'
    )
    print(
        'Ex. 3: Plot some function z(t)\n'
        't = np.arange(-3, 3, 0.01)\n'
        'z = t**3\n'
        'ax.plot(t, z, label="z(t)", linewidth=plot_size*10)\n'
        '# Set axis\n'
        'ax.axis([-2, 2, -2, 2])\n'
        '# Custom function plots() to create the plot\n'
        'plot1()\n'
        '\n'
    )
    print(
        'Ex. 4: Plot a complex function z(t)\n'
        't = np.arange(-3, 6, 0.1)\n'
        '# Use 1j for the imaginary number, not sympy.I\n'
        'z = (1-t)*(-2-1j)+t*(2+3*1j)\n'
        'ax.plot(z.real, z.imag, label="z(t)", linewidth=plot_size*10)\n'
        '# Set axis\n'
        'ax.axis([-3, 6, -3, 7])\n'
        '# Custom function plots() to create the plot\n'
        'plot1()\n'
        '\n'
    )
    print(
        "Settings and commands:\n"
        "- plot_size = .35 - Set plot size\n"
        "- ax.axis([-x, +x, -y, +y]) - Set the axis zoom. Set it optionally, by default matplotlib sets the zoom\n"
        "  fine.\n"
        "- ax.lines.pop(<graph number to remove>) - Remove a graph from the plot\n"
        "- `import numpy as np` - Use e.g. np.pi, np.sin.. with matplotlib, not sympy.\n"
        "- \n"
        "\n"
    )

def start_calculator():
    pass


# Set up plot initially
make_plot = MakePlot()
make_plot.new_plot()

ax = make_plot.ax

# Create a plot and set a new plot
plot1 = make_plot.plot1

# Plot inside existing plot
plot2 = make_plot.plot2


c = clear
h = help
hp = help_plot

# Default symbols
a = symbols('a')
b = symbols('b')
c = symbols('c')
x = symbols('x')
y = symbols('y')
z = symbols('z')
t = symbols('t')
u = symbols('u')

print("plot_calculator\n")
print("- help() - show all calculator commands\n")
print("- help_plot() to see examples for plotting graphs.\n")

code.interact(local=locals())
