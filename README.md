# KooROR
Monte-Carlo trading toolkit (Python GUI): Risk of Ruin, Expectancy, and RR-WinRate with interactive charts.  Desktop app for traders: simulate RoR, compute expectancy/Kelly, and visualize RR vs win-rate.  Tkinter + Matplotlib GUI for Risk of Ruin and expectancy analysis with zoomable plots and CSV export.

# ROR • Expectancy • RR–WinRate

*A sleek Monte-Carlo toolkit for traders — desktop GUI, zero config.*

![banner](https://placehold.co/1200x220/0e1220/e6e9ef?text=ROR+•+Expectancy+•+RR–WinRate)

---

## ✨ Features

* **Risk of Ruin (RoR)** via Monte-Carlo with fixed-fraction sizing
* **Expectancy**, **Breakeven win-rate**, and **Kelly fraction** (display-clamped)
* Plots (2×2 dashboard):

  1. **RR vs Win-Rate** with breakeven curve & your point
  2. **Expectancy heatmap** *(single persistent colorbar — no duplicates)*
  3. **Sample equity curves** (compounding)
  4. **Ending-equity distribution**
* **Scroll-wheel zoom at cursor**, plus Matplotlib toolbar (pan/zoom/save)
* **Save Charts (PNG)** and **Export CSV** (summary, ending equity, sample curves)
* **Copy Stats** button for quick sharing

---

## 📦 Prerequisites

* **Python 3.9+**
* **pip** (bundled with recent Python)
* **Tkinter** (GUI backend)

  * Windows/macOS: preinstalled with the official Python build
  * Linux:

    * Ubuntu/Debian: `sudo apt-get install -y python3-tk`
    * Fedora: `sudo dnf install python3-tkinter`
    * Arch: `sudo pacman -S tk`

---

## 🚀 Installation

**A) Quick start**

```bash
pip install -U numpy matplotlib
```

**B) (Recommended) Use a virtual environment**

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install numpy matplotlib
```

Download/save the script as **`ror_expectancy_rr_gui.py`** in your project folder.

---

## ▶️ Run

```bash
python ror_expectancy_rr_gui.py
```

---

## 🧭 Usage Guide

### Inputs (left panel)

| Field                           | Meaning                                          |
| ------------------------------- | ------------------------------------------------ |
| **Starting Capital**            | Initial equity (your base currency)              |
| **Risk per Trade (%)**          | Fixed fraction of **current** equity (compounds) |
| **Win Rate (%)**                | Probability of a winning trade                   |
| **Reward : Risk (RR)**          | Example: `1.5` → +1.5R on wins, −1R on losses    |
| **Trades (N)**                  | Number of trades per simulated path              |
| **Ruin Threshold (% drawdown)** | `100` = true ruin; `50` = ruin at −50% DD        |
| **Simulations**                 | Number of Monte-Carlo paths                      |
| **Random Seed**                 | Set for reproducible results (or leave blank)    |

### Controls

* **Update Charts** — recompute & redraw the dashboard
* **Reset Defaults** — restore safe defaults
* **Copy Stats** — copies key metrics to clipboard
* **File → Save Charts (PNG)** — exports the 2×2 figure
* **File → Export CSV** — summary + ending-equity series + sample paths

### Chart interaction

* **Mouse wheel** → zoom at cursor (per subplot)
* **Toolbar** → pan, rectangle zoom, home/back/forward, save

---

## 📐 Model & Formulas

**Trade update (fixed-fraction):**

* Win: `equity *= (1 + risk% * RR)`
* Loss: `equity *= (1 − risk%)`

**Expectancy (R/trade):**

$$
E_R = p \cdot RR - (1 - p)
$$

**Expectancy (% equity risked/trade):**

$$
E_{\%} = \text{risk\%} \times E_R
$$

**Breakeven win-rate:**

$$
p_{BE} = \frac{1}{1 + RR}
$$

**Kelly fraction (display-clamped to \[0,1]):**

$$
f^{*} = p - \frac{1-p}{RR}
$$

**Risk of Ruin:** share of paths that ever cross

$$
\text{RuinThreshold} = \text{StartingCapital}\times(1-\text{RuinDD\%})
$$

---

## 🧪 Typical Use-Cases

* Tuning **risk% / RR / win-rate** for system design
* Visualizing **tail risk** and dispersion of outcomes
* Generating **publication-ready** charts for docs, Discord, or slides

---

## ⚠️ Limitations

* Assumes **independent** trades and **stationary** p & RR
* **Fixed-fraction** compounding; **fees/slippage** not modeled
* RR treated as a single value per run (real strategies have distributions)
* Monte-Carlo has sampling error — increase `Simulations` for stability
* For research/education only — **not financial advice**

---

## 🛠️ Troubleshooting

**`_tkinter` / backend not found**
Install Tkinter (see Prerequisites). The script sets `matplotlib.use("TkAgg")`.

**Blank GUI on WSL**
Use an X server (e.g., VcXsrv) or run natively on Windows/macOS/Linux.

**Colorbar duplicates on refresh**
Already fixed: the heatmap uses a **single persistent colorbar** that updates rather than stacking.

**Cramped layout**
Resize the window or export with **File → Save Charts** (higher DPI).

---

## 📁 Project Structure

```
.
├─ ror_expectancy_rr_gui.py   # The app
└─ README.md                  # This file
```

---

## 📜 License

**MIT** — do what you want; please keep the notice.

---

## 🙌 Credits

Built with **Tkinter**, **Matplotlib**, and **NumPy**.
Design & engineering: you + ChatGPT. A star or shout-out is appreciated.
