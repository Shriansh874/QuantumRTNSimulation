# Quantum RTN Simulation

This repository contains code for simulating Random Telegraph Noise (RTN) and its effect on quantum systems. It includes the generation of RTN signals, adding noise, quantum time evolution, and various plots and statistical analyses.

## Features
- Generate RTN signals with specified high and low states.
- Add Gaussian and 1/f (flicker) noise to the signal.
- Simulate quantum time evolution using a given Hamiltonian.
- Plot signals, Power Spectral Density (PSD), Fourier Transform (FFT), autocorrelation, partial autocorrelation, and histogram of the signals.
- Locate and visualize charge traps in the signal.
- Perform statistical analysis of the noisy signal.

## Requirements
- numpy
- matplotlib
- scipy
- statsmodels

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
