from pyopenms import MSExperiment, MzMLFile
import numpy as np
import pandas as pd
import os

# Argument parser for user-defined input
import argparse

parser = argparse.ArgumentParser(description="Analyze marker intensities in mzML files.")

parser.add_argument("--file_path", required=True, help="Path to the mzML file.")
parser.add_argument("--save_path", required=True, help="Directory to save results.")
parser.add_argument("--target_marker", required=True, help="Target marker name.")
parser.add_argument("--target_mz", type=float, required=True, help="Target marker m/z value.")
parser.add_argument("--excluded_marker", required=True, help="Excluded marker name.")
parser.add_argument("--excluded_mz", type=float, required=True, help="Excluded marker m/z value.")
parser.add_argument("--excluded_tolerance", type=float, default=0.01, help="Tolerance (Da) for excluded marker.")
parser.add_argument("--tolerance", type=float, default=0.01, help="Tolerance (Da) for peak detection.")
parser.add_argument("--mz_min", type=float, default=1000, help="Minimum m/z value for binning.")
parser.add_argument("--mz_max", type=float, default=1500, help="Maximum m/z value for binning.")
parser.add_argument("--bin_width", type=float, default=0.01, help="Bin width for intensity aggregation.")
parser.add_argument("--target_sigma", type=float, default=100, help="Intensity threshold (sigma) for target marker detection.")
parser.add_argument("--excluded_sigma", type=float, default=100, help="Intensity threshold (sigma) for excluded marker detection.")

args = parser.parse_args()

# Compute detection limit
target_detection_limit = 2 * args.target_sigma
excluded_detection_limit = 2 * args.excluded_sigma


# Ensure save directory exists
os.makedirs(args.save_path, exist_ok=True)

def load_experiment(file_path):
    """Load mzML file into an MSExperiment object."""
    exp = MSExperiment()
    mzml_file = MzMLFile()
    try:
        mzml_file.load(file_path, exp)
        print(f"Data loaded from {file_path}")
    except Exception as e:
        print(f"Error loading mzML file: {e}")
    return exp

def analyze_spectra(exp):
    """Analyze spectra and classify based on target and excluded markers."""
    target_metadata_excluding_excluded = []
    target_metadata_including_excluded = []
    all_binned_intensities = []
    bins = np.arange(args.mz_min, args.mz_max + args.bin_width, args.bin_width)

    count_excluding_excluded = 0
    count_including_excluded = 0

    for spectrum_idx, spectrum in enumerate(exp.getSpectra()):
        if spectrum.getMSLevel() == 1:
            mz, intensity = spectrum.get_peaks()

            # Check if target and excluded markers are present
            target_mask = (mz >= args.target_mz - args.tolerance) & (mz <= args.target_mz + args.tolerance)
            excluded_mask = (mz >= args.excluded_mz - args.excluded_tolerance) & (mz <= args.excluded_mz + args.excluded_tolerance)

            has_target = np.any(target_mask)
            has_excluded = np.any(excluded_mask)

            if has_target:
                max_target_intensity = np.max(intensity[target_mask])
                max_excluded_intensity = np.max(intensity[excluded_mask]) if has_excluded else 0

                # Apply target marker detection limit
                if max_target_intensity > target_detection_limit:
                    # **Without excluded marker**
                    if not has_excluded or max_excluded_intensity <= excluded_detection_limit:
                        target_metadata_excluding_excluded.append({"Spectrum": spectrum_idx, "Intensity": max_target_intensity})
                        count_excluding_excluded += 1
                    
                    # **With excluded marker included**
                    target_metadata_including_excluded.append({"Spectrum": spectrum_idx, "Intensity": max_target_intensity})
                    count_including_excluded += 1

                    # Aggregate intensity data for binning
                    binned_intensities = np.zeros(len(bins) - 1)
                    indices = np.digitize(mz, bins) - 1
                    for i, idx in enumerate(indices):
                        if 0 <= idx < len(binned_intensities):
                            binned_intensities[idx] += intensity[i]
                    all_binned_intensities.append(binned_intensities)

    aggregate_intensity = np.mean(all_binned_intensities, axis=0) if all_binned_intensities else np.zeros_like(binned_intensities)

    # Normalize the aggregate intensity to a scale of 0-100
    max_aggregate_intensity = np.max(aggregate_intensity)
    if max_aggregate_intensity > 0:
        aggregate_intensity = (aggregate_intensity / max_aggregate_intensity) * 100

    avg_signal_excluding_excluded = np.mean([entry["Intensity"] for entry in target_metadata_excluding_excluded]) if target_metadata_excluding_excluded else np.nan
    avg_signal_including_excluded = np.mean([entry["Intensity"] for entry in target_metadata_including_excluded]) if target_metadata_including_excluded else np.nan

    print(f"{args.target_marker} count **without** {args.excluded_marker}: {count_excluding_excluded}")
    print(f"{args.target_marker} count **including** {args.excluded_marker}: {count_including_excluded}")
    print(f"Average {args.target_marker} Intensity (excluding {args.excluded_marker}): {avg_signal_excluding_excluded}")
    print(f"Average {args.target_marker} Intensity (including {args.excluded_marker}): {avg_signal_including_excluded}")

    return aggregate_intensity, bins[:-1], target_metadata_excluding_excluded, target_metadata_including_excluded

# Load mzML file
exp = load_experiment(args.file_path)

# Run analysis
aggregate_intensity, bins, target_metadata_excluding_excluded, target_metadata_including_excluded = analyze_spectra(exp)

# Convert lists to DataFrames
df_excluding_excluded = pd.DataFrame(target_metadata_excluding_excluded)
df_including_excluded = pd.DataFrame(target_metadata_including_excluded)

# Save CSV files
df_excluding_excluded.to_csv(os.path.join(args.save_path, f"{args.target_marker}_intensities_excluding_{args.excluded_marker}.csv"), index=False)
df_including_excluded.to_csv(os.path.join(args.save_path, f"{args.target_marker}_intensities_including_{args.excluded_marker}.csv"), index=False)

# Find spectra that included the excluded marker
set_excluding_excluded = set(df_excluding_excluded["Spectrum"])
set_including_excluded = set(df_including_excluded["Spectrum"])
overlapping_spectra = list(set_including_excluded - set_excluding_excluded)

# Save overlapping spectra
df_overlapping = pd.DataFrame({"Spectrum": overlapping_spectra})
df_overlapping.to_csv(os.path.join(args.save_path, f"{args.target_marker}_{args.excluded_marker}_overlap_spectra.csv"), index=False)

print("\nResults saved as:")
print(f" - '{args.target_marker}_intensities_excluding_{args.excluded_marker}.csv' (without {args.excluded_marker})")
print(f" - '{args.target_marker}_intensities_including_{args.excluded_marker}.csv' (with {args.excluded_marker})")
print(f" - '{args.target_marker}_{args.excluded_marker}_overlap_spectra.csv' (spectra containing both markers)")
