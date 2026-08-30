import os
from pathlib import Path
import urllib.request

import numpy as np
import pickle
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import seaborn as sns

from aiddp.styles import CRIMSON, BLUSH, BLUE, INK, BODY, MUTED, HAIR, BLUEPALE


def fetch(url: str, dest, force: bool = False) -> Path:
    """Download `url` to `dest` if not already present. Cross-platform, no shell."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest

def setup_visualization_style():
    """Configure consistent visualization style for the notebook"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(sns.color_palette('PuBu'))
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16

def setup_rdkit_drawing():
    """Configure RDKit drawing settings for consistent molecular visualizations"""
    d2d = Draw.MolDraw2DSVG(-1, -1)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    dopts.setHighlightColour((.635, .0, .145, .4))
    dopts.baseFontSize = 1.0
    dopts.additionalAtomLabelPadding = 0.15
    return dopts

def save_molecular_dataframe(df, filename, chapter="ch01", compress=True):
    """
    Save a pandas DataFrame containing molecular data to a pickle file.
    
    This function handles dataframes that contain mixed data types including
    RDKit Mol objects, which cannot be saved with standard methods like
    df.to_csv() or df.to_parquet().
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save, which may contain RDKit Mol objects
    filename : str
        Name of the file (without path)
    chapter : str
        Chapter identifier for directory organization
    compress : bool
        Whether to use compression (recommended for large dataframes)
    
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create the artifacts directory if it doesn't exist
    save_dir = Path(f"artifacts/{chapter}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Add .pkl or .pkl.gz extension if not present
    if not filename.endswith('.pkl') and not filename.endswith('.pkl.gz'):
        filename = f"{filename}.pkl"
    
    # Add compression extension if requested
    if compress and not filename.endswith('.gz'):
        filename = f"{filename}.gz"
    
    # Full path for saving
    save_path = save_dir / filename
    
    # Save the dataframe using pickle with optional compression
    protocol = pickle.HIGHEST_PROTOCOL  # Use the most efficient protocol
    
    print(f"Saving dataframe with {len(df)} rows to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(df, f, protocol=protocol)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Successfully saved dataframe ({file_size_mb:.1f} MB)")
    
    return str(save_path)

def load_molecular_dataframe(filename, chapter="ch01"):
    """
    Load a pandas DataFrame containing molecular data from a pickle file.
    
    Parameters:
    -----------
    filename : str
        Name of the file to load (without path)
    chapter : str
        Chapter identifier for directory organization
    
    Returns:
    --------
    pandas.DataFrame
        The loaded DataFrame
    """
    # Create the full file path, anchored to the repo root (the directory
    # containing utils.py) so that it works regardless of the caller's CWD.
    # The notebook kernel's CWD can land in arbitrary subdirectories (e.g.
    # data/ch02/) when cells change directories mid-run, which previously
    # caused a FileNotFoundError at runtime.
    repo_root = Path(__file__).resolve().parent
    file_dir = repo_root / "artifacts" / chapter
    
    # Handle different possible file extensions
    if not (filename.endswith('.pkl') or filename.endswith('.pkl.gz')):
        # Try both compressed and uncompressed versions
        if (file_dir / f"{filename}.pkl.gz").exists():
            filename = f"{filename}.pkl.gz"
        else:
            filename = f"{filename}.pkl"
    
    file_path = file_dir / filename
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading molecular dataframe from {file_path}...")
    start_time = pd.Timestamp.now()
    
    # Load the dataframe
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    # Calculate loading time
    load_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"Successfully loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
    print(f"Loading time: {load_time:.2f} seconds")
    
    return df

def list_saved_dataframes(chapter="ch01"):
    """
    List all saved dataframes in the artifacts directory.
    
    Parameters:
    -----------
    chapter : str
        Chapter identifier for directory organization
    
    Returns:
    --------
    list
        List of available dataframe filenames
    """
    save_dir = Path(f"artifacts/{chapter}/")
    
    if not save_dir.exists():
        print(f"No artifacts directory found for chapter {chapter}")
        return []
    
    # Get all pickle files
    saved_files = list(save_dir.glob("*.pkl*"))
    
    if not saved_files:
        print(f"No saved dataframes found in {save_dir}")
        return []
    
    # Print information about available files
    print(f"Available saved dataframes in {save_dir}:")
    file_info = []
    
    for file_path in saved_files:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        modified_time = pd.Timestamp.fromtimestamp(os.path.getmtime(file_path))
        
        file_info.append({
            'filename': file_path.name,
            'size_mb': file_size_mb,
            'modified': modified_time
        })
        
    # Sort by modification time (newest first)
    file_info.sort(key=lambda x: x['modified'], reverse=True)

    # Display the information
    for info in file_info:
        print(f"  {info['filename']} ({info['size_mb']:.1f} MB, modified: {info['modified']})")

    return [info['filename'] for info in file_info]


def set_seed(seed=42, deterministic=False):
    """
    Seed Python, NumPy, and (if installed) PyTorch + CUDA for reproducibility.

    Call once near the top of a notebook, e.g. ``SEED = set_seed(42)``. Seeding
    every RNG the notebooks touch -- plus ``PYTHONHASHSEED`` -- is what makes the
    printed outputs reproducible across runs and machines.

    Parameters
    ----------
    seed : int
        The seed applied to all random number generators.
    deterministic : bool
        If True, also force deterministic cuDNN / PyTorch kernels. This can slow
        training and some ops have no deterministic implementation, so it is
        opt-in (``warn_only=True`` avoids hard failures on those ops).

    Returns
    --------
    int
        The seed, so you can capture it in one line: ``SEED = set_seed(42)``.
    """
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed


def preview_df(df, name="df", n=3):
    """
    Print a compact before/after snapshot of a DataFrame and return its head.

    Standardizes the "show shape + columns + head after each major transform"
    pattern so readers can see exactly how the data changes from step to step
    (raw -> parsed -> descriptors -> filters -> fingerprints -> hits).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to preview.
    name : str
        A label for the printout (e.g. "raw", "filtered", "with descriptors").
    n : int
        Number of head rows to return for display.

    Returns
    --------
    pandas.DataFrame
        ``df.head(n)`` so the call renders a preview as the cell's output.
    """
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")
    return df.head(n)


def check_env(packages=None):
    """
    Print the Python version and key package versions for reproducibility.

    Several listings are version-sensitive (RDKit descriptor counts, scikit-learn
    calibration APIs, PyTorch schedulers). Printing versions up front makes a
    notebook's results self-documenting and much easier to debug across the
    local, WSL2, and Colab environments the book supports.

    Parameters
    ----------
    packages : list of str, optional
        Import names to report. Defaults to the packages the book relies on.
    """
    import importlib
    import platform

    if packages is None:
        packages = [
            "numpy", "pandas", "scipy", "sklearn", "rdkit",
            "torch", "torch_geometric", "xgboost", "matplotlib",
        ]

    print(f"Python {platform.python_version()} ({platform.system()})")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA available: no (CPU only)")
    except ImportError:
        pass

    for name in packages:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {name:16s} {version}")
        except ImportError:
            print(f"  {name:16s} (not installed)")


def get_device():
    """
    Return the best available PyTorch device, hardware-agnostic (CUDA / Apple MPS /
    other accelerator, else CPU).

    Prefers ``torch.accelerator`` (PyTorch >= 2.6), which unifies detection across
    CUDA, Apple MPS, Intel XPU, and others -- the pattern contributed by
    thomas-to-bcheme (GitHub PRs #24-28). Falls back to explicit CUDA/MPS checks on
    older PyTorch or if the accelerator query fails. Use alongside ``set_seed()`` in
    the standardized setup cell so every deep-learning chapter runs on whatever
    hardware the reader has (GPU, Mac, or CPU).

    Returns
    --------
    torch.device
        The selected device.
    """
    import torch

    # torch.accelerator (>= 2.6) unifies CUDA / MPS / XPU / ... detection.
    if hasattr(torch, "accelerator"):
        try:
            if torch.accelerator.is_available():
                return torch.device(torch.accelerator.current_accelerator())
        except Exception:
            pass

    # Fallback for older PyTorch (or if the accelerator query is unavailable).
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loglinear_fit(years, values):
    """Least-squares fit of log10(values) on year. Returns (slope, intercept, r2)."""
    obs = np.log10(values)
    slope, intercept = np.polyfit(years, obs, 1)
    pred = intercept + slope * years
    r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    return slope, intercept, r2


def plot_eroom_law(eroom, fits):
    """Plot Eroom's Law as two stacked log-scale panels and return the Figure.

    Panel A is the efficiency metric itself: annual values, a five-year centred mean
    (dashed before 1970, where the R&D denominator is reconstructed rather than
    reported), the fitted 1950-2010 trend continued past 2010 as a counterfactual,
    and the separate 2011-2025 trend. Panel B decomposes the ratio into its two
    parts, both re-indexed to 1950 = 1.
    """
    a, b, a2, b2 = fits["a"], fits["b"], fits["a2"], fits["b2"]
    halving, gap = fits["halving"], fits["gap"]

    fig = plt.figure(figsize=(7.6, 8.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.60, 1.0], left=0.128, right=0.975,
                          top=0.902, bottom=0.078, hspace=0.215)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)

    X0, X1 = 1947.5, 2028.5
    yr = eroom.year.values

    for ax in (axA, axB):
        ax.grid(True, which="major", axis="y", color=HAIR, lw=0.7, zorder=0)
        ax.grid(True, which="major", axis="x", color=HAIR, lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    # --- Panel A: the efficiency metric ------------------------------------
    axA.set_yscale("log")
    axA.set_xlim(X0, X1)
    axA.set_ylim(0.075, 165)

    roll = eroom.approvals_per_busd_rd_5yr.values
    axA.fill_between(yr, 0.075, roll, color=BLUSH, zorder=1, lw=0)

    # Uncertainty on the reconstructed pre-1970 denominator.
    pre = eroom.year <= 1969
    axA.fill_between(eroom.year[pre], eroom.approvals_per_busd_rd_lo[pre],
                     eroom.approvals_per_busd_rd_hi[pre],
                     color=CRIMSON, alpha=0.15, lw=0, zorder=2)

    # The 1950-2010 trend, then extrapolated forward as a counterfactual.
    xf = np.arange(1950, 2011)
    axA.plot(xf, 10 ** (a + b * xf), ls=(0, (6, 3)), color=INK, lw=1.4, zorder=5)
    xc = np.arange(2010, 2026)
    axA.plot(xc, 10 ** (a + b * xc), ls=(0, (1.6, 2.4)), color=INK, lw=1.2,
             alpha=0.55, zorder=5)

    # Raw annual values.
    axA.plot(yr, eroom.approvals_per_busd_rd, ls="none", marker="o", ms=3.7,
             mfc="none", mec=MUTED, mew=0.85, alpha=0.72, zorder=6)

    # The smoothed series, dashed where the denominator is reconstructed.
    m70 = yr <= 1970
    axA.plot(yr[m70], roll[m70], color=CRIMSON, lw=2.9, ls=(0, (4.5, 2.0)),
             solid_capstyle="round", zorder=8)
    axA.plot(yr[~m70 | (yr == 1970)], roll[~m70 | (yr == 1970)], color=CRIMSON,
             lw=2.9, solid_capstyle="round", zorder=8)

    # Post-2010 direction.
    xp = np.arange(2011, 2026)
    axA.plot(xp, 10 ** (a2 + b2 * xp), color=BLUE, lw=2.0, ls=(0, (5, 2.2)), zorder=7)

    # The provisional 2025 denominator.
    last = eroom[eroom.year == 2025].iloc[0]
    axA.plot([2025], [last.approvals_per_busd_rd], marker="o", ms=6.2, mfc="white",
             mec=CRIMSON, mew=1.8, zorder=9)

    axA.set_ylabel("New FDA drug approvals per US$ billion of R&D\n"
                   "(constant 2024 dollars, log scale)", labelpad=7)
    axA.yaxis.set_major_locator(
        FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
    axA.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.1f}"))
    axA.yaxis.set_minor_formatter(NullFormatter())

    # Title and subtitle are sized to fit the canvas in DejaVu Sans, the fallback
    # when TeX Gyre Heros is not installed (which is the usual case on Colab).
    hdr = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points",
               ha="left", va="bottom", annotation_clip=False)
    axA.annotate("Eroom's Law: pharmaceutical R&D efficiency, 1950–2025",
                 xytext=(0, 36), fontsize=14.0, color=INK, fontweight="bold", **hdr)
    axA.annotate("Output per research dollar halved every decade for sixty years, "
                 "then stopped falling.",
                 xytext=(0, 12), fontsize=10.6, color=BODY, **hdr)

    ann = dict(fontsize=9.6, color=BODY, ha="center",
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.75,
                               shrinkA=1, shrinkB=3))
    axA.annotate("1962: proof of efficacy required",
                 xy=(1965.5, 5.0), xytext=(1975.0, 30.0), **ann)

    # The wedge between what the old trend implied and what actually happened.
    mw = yr >= 2010
    cf = 10 ** (a + b * yr[mw])
    axA.fill_between(yr[mw], cf, roll[mw], where=roll[mw] > cf, color=BLUEPALE,
                     lw=0, zorder=3)
    axA.annotate(f"{gap:.1f}× above the\npre-2010 trend",
                 xy=(2016.0, 0.30), xytext=(2016.0, 1.9), fontsize=9.6,
                 color=BLUE, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.75,
                                 shrinkA=3, shrinkB=2))

    axA.text(1959.5, 112, "denominator\nreconstructed", fontsize=9.0, color=CRIMSON,
             ha="center", va="center", alpha=0.95, style="italic")
    axA.annotate("", xy=(1950, 88), xytext=(1969.6, 88),
                 arrowprops=dict(arrowstyle="|-|,widthA=0.32,widthB=0.32",
                                 color=CRIMSON, lw=0.95, alpha=0.75))

    # Set the halving-time label at the on-screen angle of the trend line, which
    # needs a first draw so the data-to-display transform is settled.
    fig.canvas.draw()
    p = axA.transData.transform(np.column_stack(
        [[1970, 1990], 10 ** (a + b * np.array([1970, 1990]))]))
    angle = np.degrees(np.arctan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))
    axA.text(1986.0, 2.55, f"halves every {halving:.1f} years", fontsize=10.4,
             color=INK, rotation=angle, rotation_mode="anchor", ha="center",
             va="bottom")

    legend_entries = [
        Line2D([], [], color=CRIMSON, lw=2.9, label="5-year centred mean"),
        Line2D([], [], ls="none", marker="o", ms=3.9, mfc="none", mec=MUTED,
               mew=0.85, label="annual value"),
        Line2D([], [], color=INK, ls=(0, (6, 3)), lw=1.4,
               label="trend, 1950–2010"),
        Line2D([], [], color=BLUE, ls=(0, (5, 2.2)), lw=2.0,
               label="trend, 2011–2025"),
    ]
    axA.legend(handles=legend_entries, loc="lower left", frameon=False, fontsize=9.6,
               handlelength=2.2, labelspacing=0.44, borderpad=0.1,
               bbox_to_anchor=(0.006, 0.012))

    # --- Panel B: spending against output ----------------------------------
    axB.set_yscale("log")
    axB.set_ylim(0.30, 620)
    axB.set_xlim(X0, X1)

    rd_i = eroom.rd_real_index_1950.values
    ap_i = eroom.approvals_index_1950.values

    axB.fill_between(yr, ap_i, rd_i, color=BLUSH, lw=0, zorder=1)
    axB.plot(yr[m70], rd_i[m70], color=BLUE, lw=2.5, ls=(0, (4.5, 2.0)), zorder=4)
    axB.plot(yr[~m70 | (yr == 1970)], rd_i[~m70 | (yr == 1970)], color=BLUE,
             lw=2.5, zorder=4)
    axB.plot(yr, ap_i, color=CRIMSON, lw=2.5, zorder=5)
    axB.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (1.2, 2.0)), zorder=3)

    rd_end = eroom[eroom.year == 2024].rd_real_index_1950.iloc[0]
    ap_end = eroom[eroom.year == 2024].approvals_index_1950.iloc[0]

    axB.text(2003.0, 330, "R&D spending\n(constant 2024 dollars)", fontsize=10.4,
             color=BLUE, ha="center", va="center", fontweight="bold")
    axB.text(2004.0, 0.60, "New drug approvals\n(5-year mean)", fontsize=10.4,
             color=CRIMSON, ha="center", va="center", fontweight="bold")
    axB.text(1987.0, 5.0, "the efficiency gap", fontsize=10.4, color=CRIMSON,
             ha="center", va="center", style="italic", alpha=0.85)

    axB.annotate("", xy=(2026.2, rd_end), xytext=(2026.2, ap_end),
                 arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
    axB.text(2027.0, np.sqrt(rd_end * ap_end), f"{rd_end / ap_end:.0f}×",
             fontsize=10.0, color=BODY, ha="left", va="center", rotation=90)

    axB.set_ylabel("Index, 1950 = 1 (log scale)", labelpad=7)
    axB.set_xlabel("Year", labelpad=6)
    axB.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axB.yaxis.set_minor_formatter(NullFormatter())
    axB.set_title(f"Real spending rose about {rd_end:.0f}-fold; "
                  f"output about {ap_end:.1f}-fold",
                  loc="left", pad=10, fontsize=11.8, color=INK)

    for ax in (axA, axB):
        ax.xaxis.set_major_locator(FixedLocator(np.arange(1950, 2026, 10)))
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(1950, 2026, 5)))
    plt.setp(axA.get_xticklabels(), visible=False)
    axA.tick_params(axis="x", length=0)

    return fig