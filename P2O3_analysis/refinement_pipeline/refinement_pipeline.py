from dara.refine import do_refinement_no_saving
 
import os
import re
import tempfile
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from pymatgen.core import Structure, Composition
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings

# Filter out pymatgen specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
# This section of code handles formulas/compositions

def precise_round(value, decimals=6, *, rounding=ROUND_HALF_UP):
    """Round value to `decimals` places using Decimal for precision."""
    if isinstance(value, Fraction):
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
    elif isinstance(value, Decimal):
        decimal_value = value
    else:
        decimal_value = Decimal(str(value))
    return float(decimal_value.quantize(Decimal("0." + "0" * decimals), rounding=rounding))

def parse_formula(formula_input, element_db=None):
    """
    Parse a chemical formula string into a dict of {element: quantity}.

    Supports integer, decimal, and fractional stoichiometries, e.g.:
        'NaFe1/3Cu1/3Mn1/3O2'  ->  {'Na': 1.0, 'Fe': 0.333333, 'Cu': 0.333333, ...}
    """
    pattern = r"([A-Z][a-z]*)(\d*\.?\d*/?\d*\.?\d*)"
    formula = {}

    for element, quantity_str in re.findall(pattern, formula_input):
        if element_db is not None and element not in element_db:
            print(f"Warning: Element {element} not found in database")
            return None

        if not quantity_str:
            quantity = 1.0
        elif "/" in quantity_str:
            num, den = quantity_str.split("/")
            quantity = precise_round(Fraction(int(num), int(den)))
        else:
            quantity = precise_round(float(quantity_str))

        formula[element] = precise_round(formula.get(element, 0.0) + quantity)

    return formula

def formula_to_string(formula, decimals=6, trim_ones=True):
    """
    Serialise a parsed formula dict back to a canonical string.

    Args:
        formula:   dict of {element: quantity} as returned by parse_formula.
        decimals:  significant decimal places for non-integer quantities.
        trim_ones: if True, omit the '1.0' suffix on elements with quantity 1.

    Returns:
        str, e.g. 'NaFe0.33333Cu0.33333Mn0.33333O2'
    """
    parts = []
    for element, quantity in formula.items():
        rounded = precise_round(quantity, decimals)
        # Check whether the value is a whole number
        if rounded == int(rounded):
            coeff = "" if (trim_ones and int(rounded) == 1) else str(int(rounded))
        else:
            # Format to `decimals` places, strip trailing zeros
            coeff = f"{rounded:.{decimals}f}".rstrip("0")
        parts.append(f"{element}{coeff}")
    return "".join(parts)

# This section simulates an XRD pattern from a CIF file
def pseudo_voigt(x, x0, fwhm, eta):
    """Calculates the Pseudo-Voigt profile."""
    mask = np.abs(x - x0) < (fwhm * 15)
    res = np.zeros_like(x)
    x_in = x[mask]
    
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2
    
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_in - x0) / sigma)**2)
    L = (1 / np.pi) * (gamma / ((x_in - x0)**2 + gamma**2))
    
    res[mask] = eta * L + (1 - eta) * G
    return res

def get_simulated_xrd(pattern, min_2th=10.0, max_2th=90.0, step=0.01, 
                      fwhm_base=0.25, eta=0.5, scale_max=1000.0):
    """
    Takes a pymatgen XRD pattern and returns the continuous (x, y) arrays.
    """
    num_points = int(round((max_2th - min_2th) / step)) + 1
    x_grid = np.linspace(min_2th, max_2th, num_points)
    y_grid = np.zeros_like(x_grid)
    
    lam_avg = 1.54184
    lam_ka1 = 1.54056
    lam_ka2 = 1.54439
    
    for pos_avg, I in zip(pattern.x, pattern.y):
        d = lam_avg / (2 * np.sin(np.radians(pos_avg / 2)))
        if lam_ka1 > 2 * d or lam_ka2 > 2 * d:
            continue
            
        pos1 = np.degrees(np.arcsin(lam_ka1 / (2 * d))) * 2
        pos2 = np.degrees(np.arcsin(lam_ka2 / (2 * d))) * 2
        fwhm = fwhm_base * (1 + 0.1 * np.tan(np.radians(pos1 / 2)))
        
        y_grid += I * pseudo_voigt(x_grid, pos1, fwhm, eta)
        y_grid += (I * 0.5) * pseudo_voigt(x_grid, pos2, fwhm, eta)
        
    if y_grid.max() > 0:
        y_grid = (y_grid / y_grid.max()) * scale_max
    y_grid[y_grid < 0.5] = 0.0 
    
    return x_grid, y_grid

# This section adjusts the XRD pattern
def rolling_ball_background(y_data, ball_radius, flat_value):
    # Calculates and subtracts background using a 1D rolling ball approximation.
    y = np.asarray(y_data).flatten()
    window_size = 2 * ball_radius + 1
    
    # Apply morphological opening via sequential min and max filters.
    rolled_min = ndimage.minimum_filter1d(y, size=window_size, mode='nearest')
    background = ndimage.maximum_filter1d(rolled_min, size=window_size, mode='nearest')
    
    # Subtract background, preventing negative values
    corrected = np.maximum(y - background, flat_value)
    
    return background, corrected

def interpolate_and_normalize(exp_data, reference_data, interpolation_step=0.02, normalize=False):
    min_x, max_x = min(exp_data['x']), max(exp_data['x'])
    common_x = np.arange(min_x, max_x, interpolation_step)

    y_exp_interp = np.interp(common_x, exp_data['x'], exp_data['y'])
    y_ref_interp = np.interp(common_x, reference_data['x'], reference_data['y'])

    if normalize:
        from scipy.stats import zscore
        y_exp_interp = zscore(y_exp_interp)
        y_ref_interp = zscore(y_ref_interp)

    return y_exp_interp, y_ref_interp, common_x

# This section Computes and processes the MSE calculation
def compute_mse_shift(exp_data, reference_data, interpolation_step=0.02, theta_shift_range=0.1):
    # Use normalized & interpolated data for MSE
    y_exp_interp, y_ref_interp, common_x = interpolate_and_normalize(
        exp_data, reference_data, interpolation_step, normalize=True
    )

    theta_step = interpolation_step
    index_shift_range = int(np.round(theta_shift_range / theta_step))

    mse_values, theta_shifts = [], []
    for shift in range(-index_shift_range, index_shift_range + 1):
        actual_theta_shift = shift * theta_step
        theta_shifts.append(actual_theta_shift)

        if shift >= 0:
            exp_slice, ref_slice = y_exp_interp[shift:], y_ref_interp[:len(y_exp_interp)-shift]
        else:
            exp_slice, ref_slice = y_exp_interp[:shift], y_ref_interp[-shift:]

        if len(exp_slice) > 1:
            mse = np.mean((exp_slice - ref_slice)**2)
        else:
            mse = float('inf')
        mse_values.append(mse)

    mse_values, theta_shifts = np.array(mse_values), np.array(theta_shifts)
    best_idx = np.argmin(mse_values)
    
    return theta_shifts[best_idx], mse_values[best_idx], theta_shifts, mse_values

def process_structure(file_name, directory, exp_data):
    """Helper to simulate XRD and calculate MSE."""
    try:
        path = os.path.join(directory, file_name)
        structure = Structure.from_file(path)

        # Space group symbol (e.g. "R-3m") via symmetry analysis
        try:
            sga = SpacegroupAnalyzer(structure)
            space_group = sga.get_space_group_symbol()
        except Exception as sg_err:
            print(f"Could not determine space group for {file_name}: {sg_err}")
            space_group = None

        # XRD Simulation
        calc = XRDCalculator()
        pattern = calc.get_pattern(structure)
        x, y = get_simulated_xrd(pattern, scale_max=1000)

        # MSE Calculation
        reference_data = {"x": x, "y": y}
        shift, mse_value, _, _ = compute_mse_shift(exp_data, reference_data)

        return {
            'reference_file': file_name,
            'space_group': space_group,
            'mse_shift': shift,
            'mse_value': mse_value,
        }
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return {
            'reference_file': file_name,
            'space_group': None,
            'mse_shift': np.nan,
            'mse_value': np.nan,
        }
    
def gather_mse_results(ref_dir, req_files_prefix, exp_data, comp_elements):
    all_results = []

    for f in os.listdir(ref_dir):
        if not f.endswith(".cif"):
            continue
            
        # Case 1: Explicitly requested prefixes
        if f.startswith(req_files_prefix):
            result = process_structure(f, ref_dir, exp_data)
            result['multiplied_val'] = result['mse_value'] + np.abs(result['mse_shift'])
            all_results.append(result)
            
        # Case 2: Subset of the elements in your formula
        else:
            # Load briefly just to check elements
            temp_struct = Structure.from_file(os.path.join(ref_dir, f))
            elements_in_cif = set(temp_struct.symbol_set)
            
            if elements_in_cif.issubset(set(comp_elements)):
                result = process_structure(f, ref_dir, exp_data)
                result['multiplied_val'] = result['mse_value'] + np.abs(result['mse_shift'])
                all_results.append(result)

    # Create and clean the final DataFrame
    results_df = (pd.DataFrame(all_results)
                .dropna(subset=['multiplied_val'])
                .sort_values('multiplied_val')
                .reset_index(drop=True)
                .round({'mse_value': 3, 'multiplied_val': 3}))
    
    return results_df

# This section needs work adjust composition code here.
def get_corrected_composition(struct, tol=0.03):
    """
    Correct a Structure's composition for refinement noise in site
    occupancies (e.g. 8.004 instead of 8), so reduced_composition/weight
    come out right.

    If any element's raw amount deviates from its nearest integer by more
    than `tol` (as a fraction of that integer), the composition is left
    unrounded - that level of deviation likely means genuine
    non-stoichiometry, not refinement noise, and shouldn't be silently
    forced to a clean ratio.

    Returns (composition_to_use, was_rounded, max_deviation).
    """
    raw_amounts = struct.composition.get_el_amt_dict()
    rounded_amounts = {el: round(amt) for el, amt in raw_amounts.items()}

    deviations = {
        el: abs(amt - rounded_amounts[el]) / max(rounded_amounts[el], 1)
        for el, amt in raw_amounts.items()
    }
    max_deviation = max(deviations.values())

    if max_deviation <= tol and all(v > 0 for v in rounded_amounts.values()):
        return Composition(rounded_amounts), True, max_deviation
    else:
        return struct.composition, False, max_deviation

def parse_element_counts(name, target_elements):
    """Extract counts of target metals from a phase name."""
    counts = {}
    for metal in target_elements:
        match = re.search(rf'{metal}(\d+\.?\d*)?', name)
        if match:
            counts[metal] = float(match.group(1)) if match.group(1) else 1.0
    return counts

def impurity_seperation(refinement_results, pure_phase_prefixes):
    # Separate impurity and pure phase entries
    impurity_entries = {
        key: value for key, value in refinement_results.items()
        if not any(key.startswith(p) for p in pure_phase_prefixes)
    }

    # Total impurity weight fraction
    impurity_weight_fraction = sum(impurity_entries.values())
    impurity_percentage = round(impurity_weight_fraction * 100, 2)
    
    return impurity_entries, impurity_percentage

def calculate_impurity_moles(pure_composition, excess_elements, impurity_entries,
                              ref_dir, comp_tol=0.03, verbose=False):
    target_elements = {
        str(el) for el in pure_composition.elements
        if str(el) not in excess_elements
    }

    pure_mw = pure_composition.weight
    impurity_totals = defaultdict(float)

    if verbose:
        print(f"[impurity_moles] pure composition: {pure_composition.reduced_formula} "
              f"(MW={pure_mw:.4f}), target elements: {sorted(target_elements)}")
        print(f"[impurity_moles] {len(impurity_entries)} impurity phase(s) to process\n")

    for phase_name, wt_frac in impurity_entries.items():
        el_counts = parse_element_counts(phase_name, target_elements)
        if not el_counts:
            if verbose:
                print(f"  - {phase_name}: no target elements found in name, skipping")
            continue

        cif_path = os.path.join(ref_dir, phase_name + ".cif")
        struct = Structure.from_file(cif_path)
        corrected_comp, was_rounded, max_deviation = get_corrected_composition(struct, tol=comp_tol)

        mw = corrected_comp.reduced_composition.weight
        scale = wt_frac * pure_mw / mw

        if verbose:
            print(f"  - {phase_name}")
            print(f"      wt_frac        = {wt_frac:.6f}")
            print(f"      parsed counts  = {el_counts}")
            print(f"      raw comp       = {struct.composition.formula}")
            print(f"      corrected comp = {corrected_comp.reduced_formula} "
                  f"(rounded={was_rounded}, max_dev={max_deviation:.4f})")
            print(f"      MW used        = {mw:.4f}")
            print(f"      scale factor   = wt_frac * pure_mw / mw = "
                  f"{wt_frac:.4f} * {pure_mw:.4f} / {mw:.4f} = {scale:.6f}")

        for el, count in el_counts.items():
            contribution = scale * count
            impurity_totals[el] += contribution
            if verbose:
                print(f"      + {el}: {scale:.6f} * {count} = {contribution:.6f} "
                      f"(running total: {impurity_totals[el]:.6f})")
        if verbose:
            print()

    return impurity_totals

def calculate_adjusted_composition(comp_amounts, impurity_moles, excess_elements,
                                    tm_normalise, decimal_places, verbose=False):
    tm_amounts, excess_amounts = {}, {}

    if verbose:
        print(f"[adjusted_composition] input moles: {comp_amounts}")
        print(f"[adjusted_composition] impurity moles: "
              f"{ {k: v for k, v in impurity_moles.items()} }\n")

    for metal, input_moles in comp_amounts.items():
        imp = impurity_moles.get(metal, 0.0)
        imp = imp if isinstance(imp, (int, float)) else 0.0

        corrected = max(input_moles - imp, 0.0)

        bucket = excess_amounts if metal in excess_elements else tm_amounts
        bucket[metal] = corrected

        if verbose:
            group = "excess" if metal in excess_elements else "TM"
            print(f"  - {metal} ({group}): {input_moles:.6f} - {imp:.6f} = {corrected:.6f}")

    tm_total = sum(tm_amounts.values())

    if verbose:
        print(f"\n[adjusted_composition] TM total (pre-normalisation): {tm_total:.6f}")

    if tm_normalise and tm_total > 0:
        tm_normalised = {m: v / tm_total for m, v in tm_amounts.items()}
        if verbose:
            print(f"[adjusted_composition] normalising TM sublattice (tm_normalise=True)")
            for m, v in tm_normalised.items():
                print(f"      {m}: {tm_amounts[m]:.6f} / {tm_total:.6f} = {v:.6f}")
    else:
        tm_normalised = tm_amounts
        if verbose and tm_normalise:
            print(f"[adjusted_composition] skipping normalisation (tm_total <= 0)")

    # Combine and order: Na first, then TM (sorted), then O
    # TO DO: USE EXCESS ELEMENTS TO DEAL WITH THIS CURRENTLY UNSURE HOW TO DO THAT
    combined = {**excess_amounts, **tm_normalised}
    ordered = {}
    for el in ["Na"] + sorted(tm_normalised) + ["O"]:
        if el in combined:
            ordered[el] = combined[el]

    if verbose:
        print(f"\n[adjusted_composition] final unrounded values: {ordered}")

    adjusted_formula = formula_to_string(ordered, decimals=decimal_places)

    if verbose:
        print(f"[adjusted_composition] adjusted formula: {adjusted_formula}")

    return adjusted_formula

def check_impurity_flags(impurity_moles, comp_amounts, how_close=0,
                          noise_tol=1e-6, verbose=False):
    """
    Flag elements whose impurity moles exceed their input moles by more
    than `how_close` (fraction of expected moles).

    `noise_tol` is a small absolute tolerance to absorb floating-point
    arithmetic noise only (not synthesis uncertainty) - set `how_close=0`
    if you want ANY exceedance beyond float noise to be flagged.

    Returns a dict of {metal: moles} for flagged elements only.
    """
    flagged = {}

    for metal, moles in impurity_moles.items():
        expected = comp_amounts[metal]
        raw_diff = moles - expected

        # Absorb float noise only - not a synthesis-tolerance decision
        exceeds = raw_diff > noise_tol

        if exceeds:
            pct_over = raw_diff / expected if expected else float('inf')
        else:
            pct_over = 0.0

        is_flagged = exceeds and pct_over > how_close

        if is_flagged:
            flagged[metal] = moles
            if verbose:
                print(f"  ! {metal}: FLAGGED - {moles:.4f} exceeds expected {expected:.4f} "
                      f"by {pct_over * 100:.1f}% (threshold {how_close * 100:.1f}%)")
        elif exceeds:
            if verbose:
                print(f"    {metal}: exceeds ({moles:.4f} vs {expected:.4f}) but within "
                      f"{how_close * 100:.1f}% tolerance ({pct_over * 100:.1f}% over)")
        else:
            if verbose:
                print(f"    {metal}: OK ({moles:.4f} vs {expected:.4f})")

    if verbose:
        if flagged:
            print(f"\n[check_impurity_flags] flagged: {list(flagged.keys())}")
        else:
            print(f"\n[check_impurity_flags] no elements flagged")

    return flagged

def load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column .xy file into separate 2theta and intensity arrays."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]

def save_xy(path: str, two_theta: np.ndarray, intensity: np.ndarray) -> None:
    """Write two_theta and intensity arrays back to a .xy file."""
    np.savetxt(path, np.column_stack([two_theta, intensity]), fmt="%.6f")

def correct_background(exp_data_path, ball_radius, flat_value):
    """
    Load an .xy file, apply rolling ball background correction,
    and write the corrected data to a temporary file.

    Returns the path to the temporary file. The caller is responsible
    for deleting it (or use as a context manager via tempfile directly).
    """
    two_theta, intensity = load_xy(exp_data_path)
    _, corrected = rolling_ball_background(intensity, ball_radius, flat_value)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".xy", delete=False, mode="w", dir=tempfile.gettempdir()
    )
    save_xy(tmp.name, two_theta, corrected)
    tmp.close()
    return tmp.name

def select_ref_phases_by_spacegroup(results_df, threshold_for_multi_val, ref_dir, dedupe_by_spacegroup=True, verbose=False):
    """
    From MSE-ranked candidates, pick the lowest-MSE CIF per unique
    space group, skipping any CIF whose space group is already
    represented by a better-scoring match.
    """
    candidates = (
        results_df[results_df['multiplied_val'] < threshold_for_multi_val]
        .sort_values('multiplied_val')
    )

    if verbose:
        print(f"[select_ref_phases] {len(candidates)} candidate(s) under threshold "
              f"({threshold_for_multi_val}), dedupe_by_spacegroup={dedupe_by_spacegroup}")

    if dedupe_by_spacegroup == False:
        if verbose:
            for fname in candidates['reference_file']:
                print(f"    + {fname}")
        return [os.path.join(ref_dir, fname) for fname in candidates['reference_file']]

    seen_space_groups = set()
    selected_files = []

    for _, row in candidates.iterrows():
        sg = row['space_group']
        fname = row['reference_file']

        # If space group is missing/unknown, don't dedupe on it -
        # just include the file so we don't silently drop it.
        if sg is None:
            selected_files.append(fname)
            if verbose:
                print(f"    + {fname} (space group unknown, included by default)")
            continue

        if sg in seen_space_groups:
            if verbose:
                print(f"    - skipping {fname} ({sg}) - space group already covered")
            continue

        seen_space_groups.add(sg)
        selected_files.append(fname)
        if verbose:
            print(f"    + {fname} ({sg})")

    if verbose:
        print(f"[select_ref_phases] selected {len(selected_files)} phase(s)\n")

    return [os.path.join(ref_dir, fname) for fname in selected_files]

def calculate_p2_o3_ratio(refinement_results, p_type_prefixes, o_type_prefixes):
    p_weights = {key: value for key, value in refinement_results.items()
        if any(key.startswith(p) for p in p_type_prefixes)}
    
    o_weights = {key: value for key, value in refinement_results.items()
        if any(key.startswith(p) for p in o_type_prefixes)}
    
    p_total = sum(p_weights.values())
    o_total = sum(o_weights.values())

    if p_total == 0 and o_total == 0:
        ratio = -1
    elif p_total == 0:
        ratio = 0
    elif o_total == 0:
        ratio = 1
    else:
        ratio = p_total/(p_total + o_total)

    return ratio

def run_mse_refinement(df_exp_list, filename, xrd_dir,ref_dir, 
                       req_files_prefix, pure_phase_prefixes, excess_elements,
                       threshold_for_multi_val, threshold_for_impurity_significance,background_removal, dedupe_by_spacegroup,
                       decimal_places):
    
    # Get row where UID is in dataframe
    row = df_exp_list.loc[df_exp_list['UID'] == filename].iloc[0]

    # Get formula information from dataframe about UID
    chemical = row['Chemical']
    comp_amounts = parse_formula(chemical)
    comp_elements = list(comp_amounts.keys())
    original_formula_str = formula_to_string(comp_amounts)
    pure_composition = Composition(original_formula_str)

    # Get XRD data based on the UID value
    exp_data_path = os.path.join(xrd_dir, f"{filename}.xy")
    exp_data = pd.read_csv(exp_data_path, header=None, names=['x', 'y'], sep=' ')
    exp_data = exp_data.astype(float)

    minimum_value = exp_data['y'].median()

    # background correction of XRD pattern for MSE
    background, exp_data['y'] = rolling_ball_background(exp_data['y'], ball_radius=50, flat_value=0)

    # Get MSE results
    results_df = gather_mse_results(ref_dir, req_files_prefix, exp_data, comp_elements)

    ref_phases = select_ref_phases_by_spacegroup(results_df, threshold_for_multi_val, ref_dir, dedupe_by_spacegroup, verbose=False)
    # print(ref_phases)

    if background_removal:
        correct_path = correct_background(exp_data_path, ball_radius=1000, flat_value = minimum_value)
        try:
            # Step 2: Refine
            refinement = do_refinement_no_saving(
                pattern_path=correct_path,
                phases=ref_phases,
                instrument_profile="d8-fds-02-LynxEyeXE",
                wavelength="Cu",
                phase_params = {
                "lattice_range": 0.05,
                "b1": "0_0^0.005",
                },
                refinement_params={"wmin": 0, "wmax": 90},
            )
        finally:
            os.remove(correct_path)

    else:
        refinement = do_refinement_no_saving(
            pattern_path=exp_data_path,
            phases=ref_phases,
            instrument_profile="d8-fds-02-LynxEyeXE",
            wavelength="Cu",
            phase_params = {
                "lattice_range": 0.05,
                "b1": "0_0^0.005",
                },
                refinement_params={"wmin": 10, "wmax": 90},
            )
        
    refinement_results = refinement.get_phase_weights(normalize=True)

    o_type_prefixes = ("Odash3", "O3")
    p_type_prefixes = ("Pdash2", "P2")
    p2o3_ratio = calculate_p2_o3_ratio(refinement_results, p_type_prefixes, o_type_prefixes)

    rwp = refinement.lst_data.rwp

    # Calculates impurity precentage with every composition included
    impurity_entries_all, impurity_precentage = impurity_seperation(refinement_results, pure_phase_prefixes)

    # Threshold for adjusting the composition, if certain phases are small we do not include them in this case as it is a rough estimated composition.
    impurity_entries_significant = {k: v for k, v in impurity_entries_all.items() if v >= threshold_for_impurity_significance}

    impurity_moles = calculate_impurity_moles(pure_composition, excess_elements, impurity_entries_significant, ref_dir, verbose=False)

    adjusted_formula = calculate_adjusted_composition(comp_amounts,impurity_moles, excess_elements,True, decimal_places)

    flagged = check_impurity_flags(impurity_moles, comp_amounts, how_close = 0.00, verbose=False)

    result = {
        "uid": filename,
        "original_formula": original_formula_str,
        "adjusted_formula": adjusted_formula,
        "impurity_percentage": impurity_precentage,
        "p2_o3_ratio": p2o3_ratio,
        "rwp": rwp,
        "flagged": flagged,
        "background_removal": background_removal,
        "threshold_for_multi_val": threshold_for_multi_val,
        "dedupe_by_spacegroup": dedupe_by_spacegroup,
        }
    
    return result

def generate_param_attempts(thresholds):
    for dedupe in [False, True]:
        for background in [False, True]:
            for threshold in thresholds:
                yield {
                    "dedupe_by_spacegroup": dedupe,
                    "background_removal": background,
                    "threshold_for_multi_val": threshold,
                } 

def resolve_material(df_exp_list, filename, xrd_dir, ref_dir,
                      req_files_prefix, pure_phase_prefixes, excess_elements,
                      threshold_for_impurity_significance, decimal_places,
                      thresholds):
    result = None

    for params in generate_param_attempts(thresholds):
        try:
            result = run_mse_refinement(
                df_exp_list=df_exp_list,
                filename=filename,
                xrd_dir=xrd_dir,
                ref_dir=ref_dir,
                req_files_prefix=req_files_prefix,
                pure_phase_prefixes=pure_phase_prefixes,
                excess_elements=excess_elements,
                threshold_for_impurity_significance=threshold_for_impurity_significance,
                decimal_places=decimal_places,
                **params,
            )
        except Exception as e:
            print(f"  ! {filename} failed with {params}: {e}")
            continue  # try the next parameter combination

        if not result["flagged"]:
            break  # success — stop escalating

    if result is None:
        # every single attempt raised — nothing to report but the UID
        result = {
            "uid": filename,
            "original_formula": None,
            "adjusted_formula": None,
            "impurity_percentage": None,
            "p2_o3_ratio": None,
            "rwp": None,
            "flagged": None,
            "background_removal": None,
            "threshold_for_multi_val": None,
            "dedupe_by_spacegroup": None,
            "error": "all attempts raised an exception",
        }
    elif result["flagged"]:
        result["adjusted_formula"] = None

    return result