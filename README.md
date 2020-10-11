# Description of plot_calculator
Facilitates to calculate and plot with sympy, numpy, scipy and matplotlib by
using presettings. Example commands in `help()` and `help_plot()` help to recall
the right commands for calculation and plotting graphs.

# Installation with pypoetry
`poetry install`
- In Linux pypoetry installs the script to
  `~/.cache/pypoetry/virtualenvs/<name-of-venv>/bin/plot_calculator`
  
  Just run this script to start the calculator. Create an alias in bashrc for convenience.
- In Windows pypoetry installs the script to:
  `C:\Users\<username>\AppData\Local\pypoetry\Cache\virtualenv\<name-of-venv>\scripts\plot_calculator.exe`

  Just run `plot_calculator`, but I recommend to create a shortcut of `plot_calculator.exe` and
  add `wt.exe` prior to the plot_calculator.exe path in the shortcut properties. `wt.exe` is
  Windows Terminal in Windows 10 (not installed by default).

# Dev notes
- Open plots in webbrowser and copy the plot to clipboard from there. Don't use the
  "addcopyfighandler" implementation, as it causes dependency issues with pywin32
