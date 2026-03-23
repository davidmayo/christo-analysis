# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.3",
#     "pandas>=3.0.1",
# ]
# ///
import dataclasses
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.expanduser().resolve()
BASE_OUTPUT_DIR = ROOT_DIR / "output"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the files to read in
REGOLITH_1_DATA_PATH = (
    ROOT_DIR / "data/christo_thesis/03182026/Az_El_cuts-combined/data.csv"
)
REGOLITH_2_DATA_PATH = (
    ROOT_DIR / "data/christo_thesis/03182026/regolith2/raw_data/data.csv"
)
KAPTON_ONLY_DATA_PATH = (
    ROOT_DIR / "data/christo_thesis/03182026/No_Reg_Kapton_Only/raw_data/data.csv"
)
NO_KAPTON_DATA_PATH = (
    ROOT_DIR / "data/christo_thesis/03182026/No_Reg_No_Kapton_HWOnly/raw_data/data.csv"
)


# Make a little container to hold the data
@dataclasses.dataclass(kw_only=True)
class ExperimentData:
    name: str
    df: pd.DataFrame = None
    path: Path

    def __post_init__(self) -> None:
        if self.df is None:
            self.df = pd.read_csv(self.path)


# Read the data
print(f"***** LOADING DATA *****")
regolith_1_data = ExperimentData(name="regolith_1", path=REGOLITH_1_DATA_PATH)
regolith_2_data = ExperimentData(name="regolith_2", path=REGOLITH_2_DATA_PATH)
kapton_only_data = ExperimentData(name="kapton_only", path=KAPTON_ONLY_DATA_PATH)
no_kapton_data = ExperimentData(name="no_kapton", path=NO_KAPTON_DATA_PATH)

experiments = [regolith_1_data, regolith_2_data, kapton_only_data, no_kapton_data]

# Show the user what we've got
for experiment in experiments:
    print(
        f"Loaded {len(experiment.df):,} entries for {experiment.name!r} from {experiment.path}"
    )


print(f"***** MODIFYING {regolith_1_data.name} data to match shape of others *****")

# `regolith_1` has more entries, because we were originally using a tighter grid, and had to stop because of mechanical issues
# The cuts in question are
#   horizontal-20-left-to-right
#   horizontal-15-right-to-left
#   horizontal-10-left-to-right
#   horizontal-5-right-to-left   (Only partially finished)
#
# We want to deal with this (in this case) by discarding the extra points in `regolith_1` that do not have a corresponding entry in `regolith_2`.
# "corresponding entry" here means "An entry where `commanded_pan` and `commanded_tilt` are within 0.1 of each other".
TRIMMED_CUT_IDS = {
    "horizontal-20-left-to-right",
    "horizontal-15-right-to-left",
    "horizontal-10-left-to-right",
    "horizontal-5-right-to-left",
}
PAN_TILT_MATCH_TOLERANCE = 0.1


def _has_matching_pan_tilt(row: pd.Series, reference_df: pd.DataFrame) -> bool:
    pan_matches = (
        reference_df["commanded_pan"] - row["commanded_pan"]
    ).abs() <= PAN_TILT_MATCH_TOLERANCE
    tilt_matches = (
        reference_df["commanded_tilt"] - row["commanded_tilt"]
    ).abs() <= PAN_TILT_MATCH_TOLERANCE
    return (pan_matches & tilt_matches).any()


_trimmed_regolith_1_indices: list[int] = []
_removed_counts_by_cut: dict[str, int] = {}

for cut_id in sorted(TRIMMED_CUT_IDS):
    cut_id: str
    _regolith_1_cut_df = regolith_1_data.df.loc[regolith_1_data.df["cut_id"] == cut_id]
    _regolith_2_cut_df = regolith_2_data.df.loc[regolith_2_data.df["cut_id"] == cut_id]

    matching_mask = _regolith_1_cut_df.apply(
        _has_matching_pan_tilt,
        axis=1,
        reference_df=_regolith_2_cut_df,
    )
    _trimmed_regolith_1_indices.extend(_regolith_1_cut_df.index[matching_mask].tolist())
    _removed_counts_by_cut[cut_id] = int((~matching_mask).sum())

regolith_1_data.df = regolith_1_data.df.loc[
    (~regolith_1_data.df["cut_id"].isin(TRIMMED_CUT_IDS))
    | regolith_1_data.df.index.isin(_trimmed_regolith_1_indices)
].copy()

# print("\nTrimmed unmatched extra points from regolith_1:")
# for cut_id in sorted(TRIMMED_CUT_IDS):
#     print(f"  {cut_id}: removed {removed_counts_by_cut[cut_id]:,} rows")
# print(f"  regolith_1 total after trimming: {len(regolith_1_data.df):,} rows")


# There are also now duplicate rows in cut "horizontal-5-right-to-left" ("duplicate" defined with the same tolerance logic as above).
# Remove the duplicates, keeping only the first instance of that commanded_pan/commanded_tilt.
_DUPLICATE_CUT_ID = "horizontal-5-right-to-left"

_duplicate_cut_df = regolith_1_data.df.loc[
    regolith_1_data.df["cut_id"] == _DUPLICATE_CUT_ID
]
_kept_duplicate_cut_indices: list[int] = []

for _row_index, _row in _duplicate_cut_df.iterrows():
    is_duplicate = False

    for kept_index in _kept_duplicate_cut_indices:
        kept_row = _duplicate_cut_df.loc[kept_index]
        pan_matches = (
            abs(kept_row["commanded_pan"] - _row["commanded_pan"])
            <= PAN_TILT_MATCH_TOLERANCE
        )
        tilt_matches = (
            abs(kept_row["commanded_tilt"] - _row["commanded_tilt"])
            <= PAN_TILT_MATCH_TOLERANCE
        )
        if pan_matches and tilt_matches:
            is_duplicate = True
            break

    if not is_duplicate:
        _kept_duplicate_cut_indices.append(_row_index)

_duplicate_rows_removed = len(_duplicate_cut_df) - len(_kept_duplicate_cut_indices)
regolith_1_data.df = regolith_1_data.df.loc[
    (regolith_1_data.df["cut_id"] != _DUPLICATE_CUT_ID)
    | regolith_1_data.df.index.isin(_kept_duplicate_cut_indices)
].copy()

print(
    f"\nRemoved {_duplicate_rows_removed:,} duplicate rows from {_DUPLICATE_CUT_ID!r}; "
    f"kept {len(_kept_duplicate_cut_indices):,} unique commanded pan/tilt points"
)

# Show the user what we've got
for experiment in experiments:
    print(
        f"Found {len(experiment.df):,} entries for {experiment.name!r} from {experiment.path}"
    )


# Make the plots
print(f"***** MAKING PLOTS *****")
cut_ids = list(regolith_1_data.df["cut_id"].unique())

for cut_id in cut_ids:
    if "horizontal" in cut_id:
        is_horizontal = True
        moving_name = "pan"
        fixed_name = "tilt"
    elif "vertical" in cut_id:
        is_horizontal = False
        moving_name = "tilt"
        fixed_name = "pan"
    else:
        raise AssertionError(f"Bad {cut_id=}")
    print(
        f"Making graphs for cut {cut_id!r}, which is {'HORIZONTAL' if is_horizontal else 'VERTICAL'}"
    )

    fig_center_rect, ax_center_rect = plt.subplots(
        figsize=(10, 8), layout="constrained"
    )
    fig_peak_rect, ax_peak_rect = plt.subplots(figsize=(10, 8), layout="constrained")
    fig_center_polar, ax_center_polar = plt.subplots(
        figsize=(10, 8), layout="constrained", subplot_kw={"projection": "polar"}
    )
    fig_peak_polar, ax_peak_polar = plt.subplots(
        figsize=(10, 8), layout="constrained", subplot_kw={"projection": "polar"}
    )

    fixed_angle: float | None = None

    for experiment in experiments:
        experiment_cut_df = experiment.df[experiment.df["cut_id"] == cut_id]
        print(f"  DEBUG: {experiment.name}: {len(experiment_cut_df)} datapoints")

        xs = experiment_cut_df[f"actual_{moving_name}"].to_numpy()
        center_ys = experiment_cut_df[f"center_amplitude"].to_numpy()
        peak_ys = experiment_cut_df[f"peak_amplitude"].to_numpy()
        if fixed_angle is None:
            fixed_angle = float(
                experiment_cut_df[f"commanded_{fixed_name}"].to_numpy()[0]
            )
            print(f"    DEBUG: {fixed_angle=}")

        ax_center_rect.plot(
            xs,
            center_ys,
            label=f"{experiment.name}",
            marker=".",
        )
        ax_center_rect.set_title(
            f"Center frequency power, {fixed_angle:+0.1f}° FIXED {fixed_name.upper()}"
        )

        ax_center_polar.plot(
            np.radians(xs),
            center_ys,
            label=f"{experiment.name}",
            marker=".",
        )
        ax_center_polar.set_title(
            f"Center frequency power, {fixed_angle:+0.1f}° FIXED {fixed_name.upper()}"
        )

        ax_peak_rect.plot(
            xs,
            peak_ys,
            label=f"{experiment.name}",
            marker=".",
        )
        ax_peak_rect.set_title(
            f"Peak power, {fixed_angle:+0.1f}° FIXED {fixed_name.upper()}"
        )

        ax_peak_polar.plot(
            np.radians(xs),
            peak_ys,
            label=f"{experiment.name}",
            marker=".",
        )
        ax_peak_polar.set_title(
            f"Peak frequency power, {fixed_angle:+0.1f}° FIXED {fixed_name.upper()}"
        )

    for ax in [ax_center_rect, ax_peak_rect, ax_center_polar, ax_peak_polar]:
        is_polar = isinstance(ax, plt.PolarAxes)
        print(f"      DEBUG: {ax=} {is_polar=}")
        ax.legend()

        from matplotlib.ticker import MultipleLocator

        if not is_polar:
            ax.xaxis.set_major_locator(MultipleLocator(45))
            ax.xaxis.set_minor_locator(MultipleLocator(15))
            ax.set_xlabel(f"{moving_name.upper()} (°)")
            ax.set_ylabel(f"Received power (dBm)")

        else:
            major_degs = np.arange(0, 360, 45)
            major_labels = ["0", "+45", "+90", "+135", "+180", "-135", "-90", "-45"]

            ax.set_xticks(np.deg2rad(major_degs))
            ax.set_xticklabels(major_labels)

            ax.set_xticks(np.deg2rad(np.arange(0, 360, 15)), minor=True)

            if is_horizontal:
                ax.set_theta_direction(-1)
                ax.set_theta_zero_location("N")

        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(2))

        ax.grid(which="major", axis="both", linewidth=0.8)
        ax.grid(which="minor", axis="both", linewidth=0.3)

        ax.tick_params(axis="x", which="minor", labelbottom=False)

    BASE_OUTPUT_DIR

    output_dir = BASE_OUTPUT_DIR / cut_id
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_center_polar.savefig(output_dir / "center_polar.png", dpi=300)
    fig_center_rect.savefig(output_dir / "center_rect.png", dpi=300)
    fig_peak_polar.savefig(output_dir / "peak_polar.png", dpi=300)
    fig_peak_rect.savefig(output_dir / "peak_rect.png", dpi=300)
    print(f"Saved pics to {output_dir}")
