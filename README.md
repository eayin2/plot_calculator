# Description of plot_calculator
Calculation enviroment in Python's interactive console, facilitating calculation and plotting
with sympy, numpy, scipy and matplotlib with subjective function imports, a small wrapper around
matplotlib and examples at hand.

Use `help_calc()` and `help_plot()` within the interactive calculation console to recall the
functions and to see examples.

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
- Opening plots with webbrowser.open to copy plots easily from the browser to the clipboard. Not
  using a custom clipboard implementation from within matplotlib's plot editor, as done formerly
  with an "addcopyfighandler" function, because that causes dependency issues with pywin32 on
  Windows and isn't straightforward.
