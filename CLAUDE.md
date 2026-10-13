# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Teaching examples for a *Dinámica y Control* (dynamics and control) course, mostly Jupyter notebooks in Python, plus a few in Julia. Content, markdown and comments are in **Spanish (es-AR)**; keep new content in Spanish unless asked otherwise. There is no package, build system, test suite or linter. Each notebook stands on its own.

## Running

- Python notebooks: open with Jupyter / VS Code. The main dependencies are `numpy`, `matplotlib`, `python-control` (usually `import control as ctrl`; some notebooks use `ct` or `ctl`), `scipy` (`integrate`, `signal`, `optimize`, `fft`, `io`), `ipywidgets`, `bqplot`, `pandas`, `sympy`. There's no requirements file. Kernels are per-year conda envs (`dyc23`, `dyc24`, `dyc25`) or the generic `python3`. `jupyter` and `jupytext` aren't on the default shell PATH, so activate the matching conda env first.
- To run a notebook headless and check that it executes: `jupyter nbconvert --to notebook --execute <path.ipynb> --output <scratch path>`. Notebooks that use `ipywidgets`, `bqplot`, `threading` or `matplotlib.animation` are interactive and won't produce meaningful output this way.
- Julia: `Julia/primer_notebook_pluto.jl` is a Pluto.jl notebook (cell markers `# ╔═╡`; don't reorder or strip them). `Julia/julia_1st_example.ipynb` uses an IJulia kernel. Both use `ControlSystems.jl`.

## Layout (by course topic)

- `modelado/`: physical modeling and linearization (tank, DC motor, maglev, cruise control, point-kinetics reactor with 1 and 6 delayed-neutron groups). `modelado/motor/` includes motor identification from experimental step data in `motor_data/` (`Vin_<V>_motor.{csv,mat}`, where `0p5` means 0.5 V).
- `analisis/`: frequency response (Bode via chirp + FFT) and motor analysis.
- `espacio_estado/`: state-space design (maglev integral control, inverted pendulum with linear and nonlinear simulation, reference scaling `Nbar`, animations).
- `root_locus/`: root-locus examples.
- `problemas_sueltos/`: standalone problems and exam solutions (e.g. `finales/sol_prob1_1er_parcial_241010.ipynb`).

## Conventions

- Notebooks are committed **with outputs**, so diffs are large. When editing a notebook, keep its existing kernelspec.
- The pendulum notebooks in `espacio_estado/pendulum/` are paired with Jupytext MyST markdown (`.md`, `formats: ipynb,md:myst`). If you edit one side, keep its pair in sync (`jupytext --sync <file>`).
- The typical notebook pattern is: define parameters → nonlinear ODE (`scipy.integrate.solve_ivp`/`odeint`) → linearize around an operating point → transfer function/state-space with `python-control` → compare the linear and nonlinear responses.
- `.ipynb_checkpoints/` is ignored. `tclab/` is an empty, untracked directory.
