# Turing Machine Enigma 

This project implements a **Turing Machine simulator in Python** and uses it to run an **Enigma-like 3-rotor substitution cipher** as the TM’s “computation.” The program runs in the terminal using **curses**.

## Features
- Minimal **Turing Machine core**: tape, head, state, and `delta(state, symbol)` transition function
- **Sparse infinite tape** (dictionary of non-blank cells)
- Enigma-style crypto: **3 rotors + reflector + ring settings + plugboard**
- Visualization:
  - tape window + head caret
  - rotor windows + wheel strips
  - plugboard pairs
  - keyboard + lampboard highlights
  - per-step signal path log
  - wiring tables panel (scrollable)

## Requirements
- Python 3.9+ recommended
- Runs in a terminal that supports curses so macOS Terminal, iTerm2, Linux terminal, Windows Terminal with Python curses support

No third-party packages are required.

## Run
```bash
python3 tm_enigma.py or python tm_enigma.py
