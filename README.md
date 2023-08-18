# `plot_calculator` - Calculator with Python's interactive shell
`plot_calculator` facilitates calculation and plotting inside Python's interactive shell by
importing selective functions from sympy, numpy, scipy and matplotlib and by using a small
wrapper around matplotlib for plotting graphs.

Use `help_calc()` and `help_plot()` within the interactive calculation console to recall the
functions and to see examples.

# Installation

## (1) With virtualenv (recommended way)
```
virtualenv myenv
git clone https://github.com/eayin2/plot_calculator.git
cd plot_calculator
../myenv/bin/pip3 install .
```

Run the calculator by executing: `../myenv/bin/plot_calculator`


## (2) Alternatively install with pypoetry
Install pypoetry, see: https://github.com/python-poetry/poetry

Then:
```
git clone https://github.com/eayin2/plot_calculator.git
cd plot_calculator
poetry install
```

Note:
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
