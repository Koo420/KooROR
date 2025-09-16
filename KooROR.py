#!/usr/bin/env python3
"""
ROR • Expectancy • RR–WinRate — GUI Toolkit (Polished Dark Theme, Colorbar-Fix)
Author: You + ChatGPT

Features:
- Modern dark theme with "card" panels and accent buttons
- RR vs Win-Rate plot (breakeven curve + your point)
- Expectancy heatmap (single persistent colorbar; no duplicates)
- Monte-Carlo sample equity curves (fixed-fraction compounding)
- Ending-equity histogram + key stats
- Scroll-wheel zoom-to-cursor on any subplot
- File → Save Charts (PNG) and Export CSV

Requirements: Python 3.9+, numpy, matplotlib
pip install numpy matplotlib
"""

import sys
import math
import csv
import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Matplotlib (TkAgg)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# --------------------------- Utilities ---------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_float(var, default=0.0):
    try:
        return float(str(var.get()).strip())
    except Exception:
        return float(default)

def safe_int(var, default=0):
    try:
        return int(float(str(var.get()).strip()))
    except Exception:
        return int(default)

def expectancy_R(p, rr):
    """Expectancy in R units (per trade)."""
    return p * rr - (1.0 - p)

def breakeven_winrate(rr):
    return 1.0 / (1.0 + rr) if rr > 0 else np.nan

def kelly_fraction(p, rr):
    """Optimal Kelly fraction for RR odds; may be negative or >1, clamp later for display."""
    if rr <= 0:
        return np.nan
    q = 1.0 - p
    return p - q/rr


# --------------------------- Simulation ---------------------------

def simulate_paths(
    start_equity: float,
    risk_frac: float,
    win_rate: float,
    rr: float,
    n_trades: int,
    n_sims: int,
    ruin_dd_frac: float,
    rng: np.random.Generator,
    sample_paths: int = 30,
):
    """
    Fixed-fraction Monte Carlo:
      if win:  equity *= (1 + risk_frac * rr)
      if loss: equity *= (1 - risk_frac)

    Returns dict with:
      'equity_paths_sample'  -> (sample_paths, n_trades+1)
      'ending_equity'        -> (n_sims,)
      'ruin_prob'            -> float in [0,1]
      'ruin_mask'            -> (n_sims,) bool
      'ruin_threshold'       -> float
    """
    n_trades = int(n_trades)
    n_sims = int(n_sims)

    # Per-trade multipliers
    up = 1.0 + risk_frac * rr
    dn = 1.0 - risk_frac

    wins = rng.random((n_sims, n_trades)) < win_rate
    mults = np.where(wins, up, dn)

    eq_paths = np.empty((n_sims, n_trades + 1), dtype=np.float64)
    eq_paths[:, 0] = start_equity
    eq_paths[:, 1:] = start_equity * np.cumprod(mults, axis=1)

    ruin_threshold = start_equity * (1.0 - ruin_dd_frac)
    ruined = (eq_paths <= ruin_threshold).any(axis=1)
    ruin_prob = ruined.mean()

    # Sample a subset of paths for plotting
    if n_sims <= sample_paths:
        pick = np.arange(n_sims)
    else:
        pick = np.linspace(0, n_sims - 1, sample_paths).astype(int)
    eq_sample = eq_paths[pick]

    return {
        "equity_paths_sample": eq_sample,
        "ending_equity": eq_paths[:, -1],
        "ruin_prob": float(ruin_prob),
        "ruin_mask": ruined,
        "ruin_threshold": ruin_threshold,
    }


# --------------------------- GUI App ---------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ROR • Expectancy • RR–WinRate")
        self.geometry("1320x820")
        self.minsize(1140, 740)

        # --------- Theme palette (Dark) ----------
        self.theme = {
            "bg":        "#0e1220",
            "surface":   "#141a2c",
            "card":      "#171f34",
            "border":    "#232b43",
            "fg":        "#e6e9ef",
            "subtle":    "#a8b2d1",
            "muted":     "#7d87a5",
            "accent":    "#7aa2ff",
            "accent_hi": "#9dc0ff",
            "warn":      "#f5c06e",
            # Matplotlib
            "ax_bg":     "#10172a",
            "grid":      "#26304b",
            "spine":     "#364062",
        }

        self._build_vars()
        self._build_style()
        self._build_menu()
        self._build_layout()
        self._style_matplotlib()

        # Initial render
        self.update_all()

    # ---------------- Theme & Style ----------------

    def _build_style(self):
        self.configure(bg=self.theme["bg"])
        s = ttk.Style(self)
        # Base
        s.theme_use("default")
        s.configure(".", background=self.theme["bg"], foreground=self.theme["fg"])

        # Frames
        s.configure("TFrame", background=self.theme["bg"])
        s.configure("Card.TFrame", background=self.theme["card"], borderwidth=0)
        s.configure("Surface.TFrame", background=self.theme["surface"])

        # Labelframe as "cards"
        s.configure("Card.TLabelframe",
                    background=self.theme["card"],
                    foreground=self.theme["fg"],
                    borderwidth=1,
                    relief="solid")
        s.configure("Card.TLabelframe.Label",
                    background=self.theme["card"],
                    foreground=self.theme["fg"])
        s.configure("Inputs.TLabelframe", background=self.theme["card"])
        s.configure("Inputs.TLabelframe.Label", background=self.theme["fg"])

        # Labels
        s.configure("TLabel", background=self.theme["card"], foreground=self.theme["fg"], font=("Segoe UI", 10))
        s.configure("Header.TLabel", background=self.theme["bg"], foreground=self.theme["fg"], font=("Segoe UI Semibold", 16))
        s.configure("Subtle.TLabel", background=self.theme["card"], foreground=self.theme["subtle"])
        s.configure("Value.TLabel", background=self.theme["card"], foreground=self.theme["accent"], font=("Consolas", 11, "bold"))

        # Entries
        s.configure("TEntry",
                    fieldbackground=self.theme["surface"],
                    foreground=self.theme["fg"],
                    insertcolor=self.theme["fg"],
                    borderwidth=0)
        s.map("TEntry",
              fieldbackground=[("focus", self.theme["surface"])],
              foreground=[("disabled", self.theme["muted"])])

        # Buttons
        s.configure("TButton",
                    background=self.theme["surface"],
                    foreground=self.theme["fg"],
                    borderwidth=0,
                    padding=(12, 8))
        s.map("TButton",
              background=[("active", self.theme["border"])],
              foreground=[("active", self.theme["fg"])])

        s.configure("Accent.TButton",
                    background=self.theme["accent"],
                    foreground="#0c1222",
                    padding=(14, 9),
                    font=("Segoe UI Semibold", 10))
        s.map("Accent.TButton",
              background=[("active", self.theme["accent_hi"])],
              foreground=[("active", "#0b1020")])

        # Combobox (future use)
        s.configure("TCombobox",
                    fieldbackground=self.theme["surface"],
                    background=self.theme["surface"],
                    foreground=self.theme["fg"])

    def _style_matplotlib(self):
        matplotlib.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "figure.facecolor": self.theme["bg"],
            "axes.facecolor": self.theme["ax_bg"],
            "axes.edgecolor": self.theme["spine"],
            "axes.labelcolor": self.theme["fg"],
            "xtick.color": self.theme["subtle"],
            "ytick.color": self.theme["subtle"],
            "grid.color": self.theme["grid"],
            "text.color": self.theme["fg"],
        })

    # ---------------- Vars ----------------

    def _build_vars(self):
        self.var_capital   = tk.StringVar(value="10000")
        self.var_risk_pct  = tk.StringVar(value="1.0")
        self.var_winrate   = tk.StringVar(value="50")
        self.var_rr        = tk.StringVar(value="1.00")
        self.var_trades    = tk.StringVar(value="200")
        self.var_ruin_pct  = tk.StringVar(value="100")  # 100% = true ruin; try 50 for -50% DD
        self.var_sims      = tk.StringVar(value="5000")
        self.var_seed      = tk.StringVar(value="")

        self.var_exp_R     = tk.StringVar(value="—")
        self.var_exp_pct   = tk.StringVar(value="—")
        self.var_be_wr     = tk.StringVar(value="—")
        self.var_kelly     = tk.StringVar(value="—")
        self.var_ror       = tk.StringVar(value="—")
        self.var_end_stats = tk.StringVar(value="—")

        self.last_results = None

    # ---------------- Menu ----------------

    def _build_menu(self):
        m = tk.Menu(self)
        self.config(menu=m)

        file_m = tk.Menu(m, tearoff=0, background=self.theme["card"], foreground=self.theme["fg"])
        file_m.add_command(label="Save Charts (PNG)…", command=self.save_charts)
        file_m.add_command(label="Export CSV…", command=self.export_csv)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=file_m)

        help_m = tk.Menu(m, tearoff=0, background=self.theme["card"], foreground=self.theme["fg"])
        help_m.add_command(label="About", command=self.show_about)
        m.add_cascade(label="Help", menu=help_m)

    # ---------------- Layout ----------------

    def _build_layout(self):
        root = ttk.Frame(self, style="Surface.TFrame")
        root.pack(fill=tk.BOTH, expand=True)

        # Header bar
        header = ttk.Frame(root, style="Surface.TFrame")
        header.pack(fill=tk.X, padx=14, pady=(12, 8))
        ttk.Label(header, text="ROR • Expectancy • RR–WinRate", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Monte-Carlo • Fixed-Fraction • Zoom at Cursor", style="Subtle.TLabel").pack(side=tk.RIGHT, padx=4)

        body = ttk.Frame(root, style="Surface.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        left = ttk.Frame(body, style="Surface.TFrame")
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(body, style="Surface.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Inputs card
        inputs = ttk.Labelframe(left, text="Inputs", style="Inputs.TLabelframe", padding=12)
        inputs.pack(fill=tk.X, pady=(0, 10))

        def row(parent, r, label, var, unit=""):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(6, 4), pady=5)
            ent = ttk.Entry(parent, textvariable=var, width=14, justify="right")
            ent.grid(row=r, column=1, sticky="e", padx=4, pady=5)
            ttk.Label(parent, text=unit, style="Subtle.TLabel").grid(row=r, column=2, sticky="w", padx=(2, 6))

        row(inputs, 0, "Starting Capital", self.var_capital, "$")
        row(inputs, 1, "Risk per Trade",   self.var_risk_pct, "% equity")
        row(inputs, 2, "Win Rate",         self.var_winrate, "%")
        row(inputs, 3, "Reward : Risk",    self.var_rr, "RR")
        row(inputs, 4, "Trades (N)",       self.var_trades, "")
        row(inputs, 5, "Ruin Threshold",   self.var_ruin_pct, "% drawdown")
        row(inputs, 6, "Simulations",      self.var_sims, "")
        row(inputs, 7, "Random Seed",      self.var_seed, "")

        btns = ttk.Frame(left, style="Card.TFrame")
        btns.pack(fill=tk.X, pady=(6, 10))
        ttk.Button(btns, text="Update Charts", style="Accent.TButton", command=self.update_all).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Copy Stats", command=self.copy_stats).pack(side=tk.LEFT, padx=6)

        # --- Metrics card
        met = ttk.Labelframe(left, text="Key Metrics", style="Card.TLabelframe", padding=12)
        met.pack(fill=tk.X)

        def kv(parent, r, key, var):
            ttk.Label(parent, text=key).grid(row=r, column=0, sticky="w", padx=(6, 4), pady=5)
            ttk.Label(parent, textvariable=var, style="Value.TLabel").grid(row=r, column=1, sticky="e", padx=8, pady=5)

        kv(met, 0, "Expectancy (R / trade)", self.var_exp_R)
        kv(met, 1, "Expectancy (% risk / trade)", self.var_exp_pct)
        kv(met, 2, "Breakeven Win-Rate", self.var_be_wr)
        kv(met, 3, "Kelly Fraction (approx.)", self.var_kelly)
        kv(met, 4, "Risk of Ruin (MC)", self.var_ror)

        ttk.Label(met, text="Ending Equity Stats").grid(row=5, column=0, sticky="w", padx=(6, 4), pady=(8, 4))
        ttk.Label(met, textvariable=self.var_end_stats, style="Subtle.TLabel", wraplength=280, justify="left").grid(
            row=6, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 6)
        )

        # --- Right side: Matplotlib figure
        self.fig = Figure(figsize=(8.6, 6.2), dpi=100, layout="constrained", facecolor=self.theme["bg"])
        self.ax_rr_wr = self.fig.add_subplot(2, 2, 1)
        self.ax_heat  = self.fig.add_subplot(2, 2, 2)
        self.ax_eq    = self.fig.add_subplot(2, 2, 3)
        self.ax_hist  = self.fig.add_subplot(2, 2, 4)

        # Persistent colorbar handle for the heatmap (prevents duplicates)
        self.heat_cbar = None

        # Canvas + toolbar
        canvas_frame = ttk.Frame(right, style="Card.TFrame")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Scroll-wheel zoom to cursor
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

        # Subtle status hint
        hintbar = ttk.Frame(root, style="Surface.TFrame")
        hintbar.pack(fill=tk.X, padx=14, pady=(0, 10))
        ttk.Label(
            hintbar,
            text="Tip: Use mouse wheel to zoom at cursor • Drag with toolbar pan • File → Save Charts / Export CSV",
            style="Subtle.TLabel"
        ).pack(side=tk.LEFT)

    # ---------------- Actions ----------------

    def reset_defaults(self):
        self.var_capital.set("10000")
        self.var_risk_pct.set("1.0")
        self.var_winrate.set("50")
        self.var_rr.set("1.00")
        self.var_trades.set("200")
        self.var_ruin_pct.set("100")
        self.var_sims.set("5000")
        self.var_seed.set("")
        self.update_all()

    def copy_stats(self):
        if not self.last_results:
            return
        self.clipboard_clear()
        buf = io.StringIO()
        buf.write("ROR • Expectancy • RR–WinRate — Stats\n")
        buf.write(f"Expectancy (R/trade): {self.var_exp_R.get()}\n")
        buf.write(f"Expectancy (% risk):  {self.var_exp_pct.get()}\n")
        buf.write(f"Breakeven Win-Rate:   {self.var_be_wr.get()}\n")
        buf.write(f"Kelly Fraction:       {self.var_kelly.get()}\n")
        buf.write(f"Risk of Ruin (MC):    {self.var_ror.get()}\n")
        buf.write(f"Ending Equity:        {self.var_end_stats.get()}\n")
        self.clipboard_append(buf.getvalue())
        messagebox.showinfo("Copied", "Stats copied to clipboard.")

    def show_about(self):
        messagebox.showinfo(
            "About",
            "ROR • Expectancy • RR–WinRate\n"
            "Monte-Carlo fixed-fraction model with compounding.\n"
            "Scroll wheel = zoom at cursor. Use the toolbar to pan/zoom/save."
        )

    def read_inputs(self):
        capital   = max(0.0, safe_float(self.var_capital, 10000))
        risk_pct  = clamp(safe_float(self.var_risk_pct, 1.0), 0.0, 100.0)
        winrate   = clamp(safe_float(self.var_winrate, 50.0), 0.0, 100.0)
        rr        = max(1e-9, safe_float(self.var_rr, 1.0))
        trades    = max(1, safe_int(self.var_trades, 200))
        ruin_pct  = clamp(safe_float(self.var_ruin_pct, 100.0), 1.0, 100.0)
        sims      = max(100, safe_int(self.var_sims, 5000))
        seed_txt  = self.var_seed.get().strip()
        seed = None
        if seed_txt:
            try:
                seed = int(seed_txt)
            except Exception:
                seed = abs(hash(seed_txt)) % (2**32 - 1)

        risk_frac = risk_pct / 100.0
        p = winrate / 100.0
        ruin_dd_frac = ruin_pct / 100.0

        return capital, risk_frac, p, rr, trades, ruin_dd_frac, sims, seed

    def update_all(self):
        capital, risk_frac, p, rr, trades, ruin_dd_frac, sims, seed = self.read_inputs()

        # Derived metrics
        E_R = expectancy_R(p, rr)
        E_pct = risk_frac * E_R * 100.0
        p_be = breakeven_winrate(rr)
        kelly = kelly_fraction(p, rr)
        kelly_disp = clamp(kelly, 0.0, 1.0) if not math.isnan(kelly) else float("nan")

        # RNG & simulation
        rng = np.random.default_rng(seed)
        results = simulate_paths(
            start_equity=capital,
            risk_frac=risk_frac,
            win_rate=p,
            rr=rr,
            n_trades=trades,
            n_sims=sims,
            ruin_dd_frac=ruin_dd_frac,
            rng=rng,
            sample_paths=30,
        )
        self.last_results = results

        ruin_prob = results["ruin_prob"]
        ending = results["ending_equity"]

        # Update labels
        self.var_exp_R.set(f"{E_R:.4f} R")
        self.var_exp_pct.set(f"{E_pct:+.3f}% of equity risked")
        self.var_be_wr.set(f"{p_be*100:.2f}%")
        self.var_kelly.set("n/a" if math.isnan(kelly) else f"{kelly_disp*100:.2f}% (clamped)")
        self.var_ror.set(f"{ruin_prob*100:.2f}%")

        if ending.size:
            q10, q50, q90 = np.percentile(ending, [10, 50, 90])
            avg = ending.mean()
            self.var_end_stats.set(
                f"Avg: ${avg:,.0f} • Median: ${q50:,.0f} • P10: ${q10:,.0f} • P90: ${q90:,.0f}"
            )
        else:
            self.var_end_stats.set("—")

        # Redraw plots
        self.draw_rr_winrate(p, rr)
        self.draw_expectancy_heat(p, rr)
        self.draw_equity_curves(results["equity_paths_sample"], results["ruin_threshold"])
        self.draw_histogram(ending)

        self.canvas.draw_idle()

    # ----------------- Plots -----------------

    def _ax_style(self, ax):
        ax.set_facecolor(self.theme["ax_bg"])
        for spine in ax.spines.values():
            spine.set_color(self.theme["spine"])
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8, color=self.theme["grid"])
        ax.title.set_color(self.theme["fg"])
        ax.xaxis.label.set_color(self.theme["fg"])
        ax.yaxis.label.set_color(self.theme["fg"])

    def draw_rr_winrate(self, p, rr):
        ax = self.ax_rr_wr
        ax.clear()
        self._ax_style(ax)
        ax.set_title("RR vs Win-Rate — Breakeven Curve")
        ax.set_xlabel("Reward : Risk (RR)")
        ax.set_ylabel("Win-Rate")

        xs = np.linspace(0.1, 6.0, 800)
        ys = 1.0 / (1.0 + xs)
        ax.plot(xs, ys, lw=2.2, label="Breakeven (E=0)")

        ax.fill_between(xs, ys, 1.0, alpha=0.18, label="Positive Expectancy")
        ax.fill_between(xs, 0.0, ys, alpha=0.10, label="Negative Expectancy")

        ax.scatter([rr], [p], s=70, marker="o", zorder=5,
                   label="Your point", edgecolor=self.theme["fg"], linewidths=0.6)

        ax.set_xlim(0, 6.0)
        ax.set_ylim(0, 1.0)
        leg = ax.legend(loc="lower right", frameon=True)
        leg.get_frame().set_alpha(0.15)

    def draw_expectancy_heat(self, p, rr):
        ax = self.ax_heat
        ax.clear()
        self._ax_style(ax)
        ax.set_title("Expectancy Heatmap (R per trade)")
        ax.set_xlabel("RR")
        ax.set_ylabel("Win-Rate")

        rr_grid = np.linspace(0.1, 6.0, 120)
        p_grid  = np.linspace(0.05, 0.95, 120)
        RRg, Pg = np.meshgrid(rr_grid, p_grid)
        E = Pg * RRg - (1 - Pg)

        vmax = float(np.nanmax(np.abs(E)))
        im = ax.imshow(
            E, origin="lower",
            extent=[rr_grid.min(), rr_grid.max(), p_grid.min(), p_grid.max()],
            aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        ax.contour(RRg, Pg, E, levels=[0.0], colors="white", linewidths=1.2, linestyles="--")
        ax.scatter([rr], [p], s=70, marker="o", edgecolor="white", linewidths=0.6)

        ax.set_xlim(0, 6.0)
        ax.set_ylim(0.05, 0.95)

        # ---- Persistent colorbar (prevents stacking on refresh) ----
        if self.heat_cbar is None:
            self.heat_cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self.heat_cbar.outline.set_edgecolor(self.theme["spine"])
            self.heat_cbar.set_label("Expectancy (R)")
        else:
            try:
                self.heat_cbar.update_normal(im)
            except Exception:
                # Rarely, mpl might complain—rebuild cleanly
                try:
                    self.heat_cbar.remove()
                except Exception:
                    pass
                self.heat_cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self.heat_cbar.outline.set_edgecolor(self.theme["spine"])
                self.heat_cbar.set_label("Expectancy (R)")

    def draw_equity_curves(self, eq_paths_sample, ruin_threshold):
        ax = self.ax_eq
        ax.clear()
        self._ax_style(ax)
        ax.set_title("Sample Equity Curves")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")

        n_paths, steps = eq_paths_sample.shape
        x = np.arange(steps)

        # Slight alpha ladder for depth
        base_alpha = 0.8
        for i in range(n_paths):
            alpha = base_alpha * (0.5 + 0.5 * (i + 1) / n_paths)
            ax.plot(x, eq_paths_sample[i], lw=1.1, alpha=alpha)

        ax.axhline(ruin_threshold, linestyle="--", lw=1.6, alpha=0.9, label="Ruin Threshold")
        leg = ax.legend(loc="upper left", frameon=True)
        leg.get_frame().set_alpha(0.15)

    def draw_histogram(self, ending_equity):
        ax = self.ax_hist
        ax.clear()
        self._ax_style(ax)
        ax.set_title("Ending Equity Distribution")
        ax.set_xlabel("Ending Equity ($)")
        ax.set_ylabel("Frequency")

        if ending_equity.size:
            ax.hist(ending_equity, bins=40, alpha=0.95)
            q10, q50, q90 = np.percentile(ending_equity, [10, 50, 90])
            ax.axvline(q50, linestyle="--", lw=1.8, label=f"Median ${q50:,.0f}")
            ax.axvline(q10, linestyle=":", lw=1.4, label=f"P10 ${q10:,.0f}")
            ax.axvline(q90, linestyle=":", lw=1.4, label=f"P90 ${q90:,.0f}")
            leg = ax.legend(loc="upper right", frameon=True)
            leg.get_frame().set_alpha(0.15)

    # ----------------- Save / Export -----------------

    def save_charts(self):
        f = filedialog.asksaveasfilename(
            title="Save Charts as PNG",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialfile="ror_expectancy_rr.png"
        )
        if not f:
            return
        try:
            self.fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=self.theme["bg"])
            messagebox.showinfo("Saved", f"Charts saved:\n{f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save charts:\n{e}")

    def export_csv(self):
        if not self.last_results:
            messagebox.showwarning("No Data", "Run a simulation first.")
            return

        f = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="ror_expectancy_export.csv"
        )
        if not f:
            return

        capital, risk_frac, p, rr, trades, ruin_dd_frac, sims, seed = self.read_inputs()
        res = self.last_results
        eq_sample = res["equity_paths_sample"]

        try:
            with open(f, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["# ROR • Expectancy • RR–WinRate Export"])
                w.writerow(["StartingCapital", capital])
                w.writerow(["RiskPerTrade(%)", risk_frac * 100.0])
                w.writerow(["WinRate(%)", p * 100.0])
                w.writerow(["RR", rr])
                w.writerow(["Trades", trades])
                w.writerow(["RuinThreshold(%)", ruin_dd_frac * 100.0])
                w.writerow(["Simulations", sims])
                w.writerow(["Seed", "" if seed is None else seed])
                w.writerow([])

                # Summary
                E_R = expectancy_R(p, rr)
                E_pct = risk_frac * E_R * 100.0
                p_be = breakeven_winrate(rr) * 100.0
                kelly = kelly_fraction(p, rr)
                kelly_disp = clamp(kelly, 0.0, 1.0) * 100.0 if not math.isnan(kelly) else float("nan")
                w.writerow(["Metric", "Value"])
                w.writerow(["Expectancy (R/trade)", f"{E_R:.6f}"])
                w.writerow(["Expectancy (% risk/trade)", f"{E_pct:.6f}"])
                w.writerow(["Breakeven WinRate (%)", f"{p_be:.6f}"])
                w.writerow(["Kelly Fraction (clamped %)", "n/a" if math.isnan(kelly) else f"{kelly_disp:.6f}"])
                w.writerow(["Risk of Ruin (MC %)", f"{res['ruin_prob']*100:.6f}"])
                w.writerow([])

                # Ending equity distribution
                w.writerow(["# EndingEquity"])
                w.writerow(["EndingEquity"])
                for v in res["ending_equity"]:
                    w.writerow([f"{v:.6f}"])
                w.writerow([])

                # Sample equity paths
                w.writerow(["# SampleEquityPaths"])
                header = ["Trade#"] + [f"Path{i+1}" for i in range(eq_sample.shape[0])]
                w.writerow(header)
                for t in range(eq_sample.shape[1]):
                    row = [t] + [f"{eq_sample[i, t]:.6f}" for i in range(eq_sample.shape[0])]
                    w.writerow(row)

            messagebox.showinfo("Exported", f"CSV exported:\n{f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV:\n{e}")

    # ----------------- Mouse Wheel Zoom -----------------

    def _on_scroll_zoom(self, event):
        """Zoom the axes under the cursor, centered at cursor."""
        ax = event.inaxes
        if ax is None:
            return
        # Direction: 'up' -> zoom in, 'down' -> zoom out
        base = 1.2
        scale = 1 / base if event.button == "up" else base

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())
        xw = (xlim[1] - xlim[0]) * scale
        yw = (ylim[1] - ylim[0]) * scale

        # Relative position inside current view
        rx = (xdata - xlim[0]) / (xlim[1] - xlim[0]) if (xlim[1] - xlim[0]) != 0 else 0.5
        ry = (ydata - ylim[0]) / (ylim[1] - ylim[0]) if (ylim[1] - ylim[0]) != 0 else 0.5

        ax.set_xlim(xdata - xw * rx, xdata + xw * (1 - rx))
        ax.set_ylim(ydata - yw * ry, ydata + yw * (1 - ry))
        self.canvas.draw_idle()


# --------------------------- main ---------------------------

if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        print("Fatal error:", e, file=sys.stderr)
        sys.exit(1)
