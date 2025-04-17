# -*- coding: utf-8 -*-
"""
Combined and Refined FIFA Player Data Cleaning and Analysis Script
with Section Skipping Capability
"""
# --- Core Libraries ---
import pandas as pd
import numpy as np
import warnings

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Statistical Analysis ---
import scipy.stats as stats

# --- Machine Learning (Scikit-learn & LightGBM) ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    print("Warning: lightgbm library not found. Section 8 will be skipped.")
    print("Install it using: pip install lightgbm")
    LGBM_INSTALLED = False

# --- Configuration ---

# File path for the dataset
# !! IMPORTANT: Make sure 'fifa_players.csv' is in the same directory
# !! or provide the full path to the file.
FILE_PATH = r'C:\Users\acyp2\OneDrive\Desktop\BACS3013  DATA SCIENCE\fifa_players.csv'

# Missing Value Handling
COLUMN_DROP_THRESHOLD_PCT = 50.0 # Drop columns missing more than this %
ROW_DROP_THRESHOLD_PCT = 15.0  # Drop rows if % of rows with NaNs is below this

# Outlier Handling
OUTLIER_COLUMNS_TO_CHECK = ['wage_euro', 'value_euro', 'release_clause_euro', 'potential', 'overall_rating']
RATING_LOWER_BOUND = 40 # Assumed minimum plausible rating
RATING_UPPER_BOUND = 99 # Maximum possible rating
FINANCIAL_LOWER_BOUND = 0 # Financial values cannot be negative

# Correlation Analysis
CORRELATION_TARGET_COLUMNS = ['potential', 'wage_euro', 'value_euro', 'release_clause_euro']
CORRELATION_THRESHOLD = 0.1 # Features below this |corr| with ALL targets might be dropped
MAX_MISSING_TARGET_PERCENTAGE = 50.0 # Exclude target columns if missing > this %

# ANOVA Test
TARGET_FOR_ANOVA = 'value_euro' # Target variable for ANOVA
TOP_N_NATIONALITIES = 15 # Number of top nationalities to keep separate
ALPHA_LEVEL = 0.05 # Significance level for p-value

# RandomForest Feature Importance & Linear Regression
TARGET_FOR_RF = 'value_euro' # Also used by LR section
CORE_PREDICTORS_RF = ['overall_rating', 'potential', 'age'] # Also used by LR section
CATEGORICAL_FEATURE_RF = 'nationality' # Also used by LR section
encoded_feature_name_rf = f'{CATEGORICAL_FEATURE_RF}_freq_encoded' # Name for the new column
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
RF_N_JOBS = -1


# --- !! Section Execution Control !! ---
# Set these flags to True to run the corresponding section, False to skip.
# NOTE: Skipping cleaning steps (1, 2) might affect the results of later analyses.
#       Sections 5 & 7 perform their own NaN handling for required columns.
RUN_SECTION_1_MISSING_VALUES = False  # Set to True to run Missing Value Handling
RUN_SECTION_2_OUTLIERS = False        # Set to True to run Outlier Handling
RUN_SECTION_3_CORRELATION = False     # Set to True to run Correlation Analysis
RUN_SECTION_4_ANOVA = False           # Set to True to run ANOVA
RUN_SECTION_5_RANDOM_FOREST = False   # Set to True to run RandomForest Importance
RUN_SECTION_7_LINEAR_REGRESSION = False # Set to True to run Linear Regression
RUN_SECTION_8_LIGHTGBM = True # Set to True to run LightGBM (Light Gradient Boosting Machine)

# --- Settings ---
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore some seaborn/pandas warnings
pd.set_option('display.max_columns', None) # Show all columns in printouts
pd.set_option('display.width', 1000) # Wider display for printouts
plt.style.use('seaborn-v0_8-whitegrid') # Set a default plot style


# =============================================================================
# 0. Load Data (This section always runs)
# =============================================================================
print("--- 0. Loading Data ---")
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded data from '{FILE_PATH}'.")
    print(f"Initial DataFrame shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at '{FILE_PATH}'. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

if df.empty:
    print("Error: The loaded DataFrame is empty. Exiting.")
    exit()

original_rows, original_cols = df.shape
df_cleaned = df.copy() # Start with a fresh copy


# =============================================================================
# 1. Missing Value Handling
# =============================================================================
if RUN_SECTION_1_MISSING_VALUES:
    print("\n--- [EXECUTING] 1. Handling Missing Values ---")

    # --- Step 1.1: Initial Assessment ---
    print("\n--- Step 1.1: Initial Missing Value Assessment ---")
    missing_values = df_cleaned.isnull().sum()
    missing_percent = (missing_values / original_rows) * 100
    missing_stats = pd.DataFrame({'Missing Count': missing_values, 'Missing Percent': missing_percent})
    missing_stats = missing_stats[missing_stats['Missing Count'] > 0].sort_values(by='Missing Percent', ascending=False)

    if missing_stats.empty:
        print("No missing values found in the original DataFrame.")
    else:
        print("Columns with missing values (Sorted by percentage):")
        print(missing_stats)

    # --- Step 1.2: Drop Columns with High Missing Percentage ---
    print(f"\n--- Step 1.2: Dropping Columns Missing > {COLUMN_DROP_THRESHOLD_PCT}% ---")
    cols_to_drop_missing = missing_stats[missing_stats['Missing Percent'] > COLUMN_DROP_THRESHOLD_PCT].index.tolist()

    if not cols_to_drop_missing:
        print("No columns exceed the missing value threshold for dropping.")
    else:
        print(f"Columns to drop ({len(cols_to_drop_missing)}): {cols_to_drop_missing}")
        df_cleaned = df_cleaned.drop(columns=cols_to_drop_missing)
        print(f"DataFrame shape after dropping columns: {df_cleaned.shape}")

    # --- Step 1.3: Assess Impact of Dropping Rows ---
    print("\n--- Step 1.3: Assess Impact of Dropping Remaining Rows with Missing Values ---")
    rows_before_assessment = df_cleaned.shape[0]
    rows_with_nan = df_cleaned[df_cleaned.isnull().any(axis=1)]
    num_rows_with_nan = len(rows_with_nan)

    if rows_before_assessment == 0:
        percent_rows_to_drop = 0
        print("DataFrame is empty after column drops. No rows to assess.")
    elif num_rows_with_nan == 0:
        percent_rows_to_drop = 0
        print("No rows with missing values found after column drops.")
    else:
        percent_rows_to_drop = (num_rows_with_nan / rows_before_assessment) * 100
        print(f"Rows currently in DataFrame: {rows_before_assessment}")
        print(f"Number of rows with AT LEAST ONE missing value: {num_rows_with_nan}")
        print(f"Percentage of current rows that would be dropped: {percent_rows_to_drop:.2f}%")

    # --- Step 1.4: Conditional Row Drop ---
    print(f"\n--- Step 1.4: Conditional Drop of Rows with Any Missing Value (Threshold: < {ROW_DROP_THRESHOLD_PCT}%) ---")
    rows_dropped_step1 = 0
    proceed_with_row_drop = False

    if num_rows_with_nan == 0:
        print("No rows to drop. DataFrame is already complete in terms of missing values.")

    elif percent_rows_to_drop < ROW_DROP_THRESHOLD_PCT:
        print(f"Percentage of rows with missing values ({percent_rows_to_drop:.2f}%) is below threshold ({ROW_DROP_THRESHOLD_PCT}%).")
        print(f"Analyzing characteristics of the {num_rows_with_nan} rows proposed for dropping...")

        # --- Analysis: Compare Rows to Drop vs. Rows to Keep ---
        rows_to_drop_analysis = df_cleaned[df_cleaned.isnull().any(axis=1)]
        rows_to_keep_analysis = df_cleaned.dropna()

        if rows_to_keep_analysis.empty or rows_to_drop_analysis.empty:
             print("Warning: Cannot perform comparison analysis as one group (keep/drop) is empty.")
             proceed_with_row_drop = True # Defaulting to proceed based on percentage
        else:
            numerical_compare_cols = [col for col in ['overall_rating', 'potential', 'age', 'value_euro', 'wage_euro'] if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col])]
            categorical_compare_cols = [col for col in ['preferred_foot', 'work_rate'] if col in df_cleaned.columns]

            print("\n  Comparison of Numerical Features (Mean / Median):")
            comparison_data = {}
            for col in numerical_compare_cols:
                 # Ensure data is numeric before calculating mean/median
                 mean_keep = pd.to_numeric(rows_to_keep_analysis[col], errors='coerce').mean()
                 median_keep = pd.to_numeric(rows_to_keep_analysis[col], errors='coerce').median()
                 mean_drop = pd.to_numeric(rows_to_drop_analysis[col], errors='coerce').mean()
                 median_drop = pd.to_numeric(rows_to_drop_analysis[col], errors='coerce').median()
                 comparison_data[col] = {
                     'Mean (Keep)': mean_keep, 'Mean (Drop)': mean_drop,
                     'Median (Keep)': median_keep, 'Median (Drop)': median_drop
                 }
            comparison_df_num = pd.DataFrame(comparison_data).T

            try:
                 comparison_df_num['Mean % Diff'] = ((comparison_df_num['Mean (Drop)'] - comparison_df_num['Mean (Keep)']) / comparison_df_num['Mean (Keep)']) * 100
                 comparison_df_num['Median % Diff'] = ((comparison_df_num['Median (Drop)'] - comparison_df_num['Median (Keep)']) / comparison_df_num['Median (Keep)']) * 100
                 comparison_df_num.fillna(0, inplace=True) # Replace NaN % diffs (e.g., if keep mean is 0)
            except ZeroDivisionError:
                 print("  (Skipping % difference calculation due to zero mean/median in 'Keep' group for some columns)")

            print(comparison_df_num.round(2))

            print("\n  Comparison of Categorical Features (Proportions):")
            significant_categorical_diff = False
            for col in categorical_compare_cols:
                print(f"\n    --- {col} ---")
                prop_keep = rows_to_keep_analysis[col].value_counts(normalize=True).round(3)
                prop_drop = rows_to_drop_analysis[col].value_counts(normalize=True).round(3)
                comparison_df_cat = pd.DataFrame({'Proportion (Keep)': prop_keep, 'Proportion (Drop)': prop_drop}).fillna(0)
                print(comparison_df_cat)
                if not comparison_df_cat.empty:
                     diff = abs(comparison_df_cat.iloc[0]['Proportion (Keep)'] - comparison_df_cat.iloc[0]['Proportion (Drop)'])
                     if diff > 0.10:
                          significant_categorical_diff = True
                          print(f"    *Note: Significant difference observed in '{col}' distribution.*")

            # --- Conclusion from Analysis ---
            print("\n  Analysis Conclusion:")
            large_num_diff = False
            if 'Mean % Diff' in comparison_df_num.columns:
                 key_metrics_diff = comparison_df_num.loc[comparison_df_num.index.intersection(['overall_rating', 'potential', 'value_euro']), 'Mean % Diff'].abs()
                 if (key_metrics_diff > 15).any(): # Using 15% as threshold for "large" difference
                      large_num_diff = True

            if large_num_diff or significant_categorical_diff:
                print("  - Potential systematic differences observed. Dropping these rows might introduce bias.")
                print("  - Recommendation: Consider IMPUTATION instead of dropping.")
                proceed_with_row_drop = False
            else:
                print("  - No major systematic differences detected. Proceeding with dropping rows seems reasonable.")
                proceed_with_row_drop = True

        if proceed_with_row_drop:
            print(f"\nProceeding to drop {num_rows_with_nan} rows with missing values.")
            df_cleaned = df_cleaned.dropna()
            rows_dropped_step1 = num_rows_with_nan
            print(f"DataFrame shape after dropping rows: {df_cleaned.shape}")
        else:
            print("\nSkipping row drop based on analysis or threshold. IMPUTATION recommended for remaining NaNs.")
            # If imputation was desired, it would be implemented here.
            # For now, we stop, leaving NaNs if row drop was skipped.

    elif num_rows_with_nan > 0: # Handles the case where percent_rows_to_drop >= threshold
        print(f"Percentage ({percent_rows_to_drop:.2f}%) is >= threshold ({ROW_DROP_THRESHOLD_PCT}%).")
        print("Dropping these rows would remove too much data.")
        print("IMPUTATION is strongly recommended for remaining missing values.")
        # If imputation was desired, it would be implemented here.
        # For now, we stop, leaving NaNs.

    # --- Step 1.5: Final Assessment ---
    print("\n--- Step 1.5: Final Missing Value Assessment ---")
    final_missing_values = df_cleaned.isnull().sum()
    final_missing_stats = final_missing_values[final_missing_values > 0]

    if final_missing_stats.empty:
        print("DataFrame now has NO missing values.")
    else:
        print("WARNING: Missing values still remain (due to skipped row drop or failed imputation).")
        remaining_stats_df = pd.DataFrame({
            'Missing Count': final_missing_stats,
            'Missing Percent': (final_missing_stats / df_cleaned.shape[0]) * 100
        }).sort_values(by='Missing Percent', ascending=False)
        print(remaining_stats_df)
        print("Consider imputation for these columns before proceeding with analyses that require complete data.")

else:
    print("\n--- [SKIPPING] 1. Handling Missing Values ---")


# =============================================================================
# 2. Outlier Handling
# =============================================================================
if RUN_SECTION_2_OUTLIERS:
    print("\n--- [EXECUTING] 2. Handling Outliers ---")

    # --- Check if df_cleaned has data ---
    if df_cleaned.empty:
        print("DataFrame is empty after missing value handling. Skipping outlier detection.")
    else:
        all_potential_outlier_indices = set()
        indices_to_remove_outliers = set()

        # --- Step 2.1: Identify Potential Outliers using 1.5*IQR ---
        print("\n--- Step 2.1: Identifying Potential Outliers (1.5*IQR Rule) ---")
        for column in OUTLIER_COLUMNS_TO_CHECK:
            print(f"\n--- Identifying potential outliers in: {column} ---")
            if column not in df_cleaned.columns:
                print(f"Skipping: Column '{column}' not found.")
                continue
            if not pd.api.types.is_numeric_dtype(df_cleaned[column]):
                print(f"Skipping: Column '{column}' is not numeric.")
                continue
            # Ensure column has no NaNs for IQR calculation, otherwise skip
            if df_cleaned[column].isnull().any():
                 print(f"Skipping: Column '{column}' contains NaNs. Cannot calculate IQR reliably without imputation.")
                 continue

            # Generate Boxplot (Optional but useful visualization)
            try:
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=df_cleaned[column])
                plt.title(f'Boxplot of {column}')
                plt.ylabel(column)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.show()
            except Exception as e:
                print(f"  Warning: Could not generate boxplot for {column}: {e}")

            # Calculate bounds
            try:
                Q1 = df_cleaned[column].quantile(0.25)
                Q3 = df_cleaned[column].quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0:
                    print("IQR is zero. No outliers defined by this rule.")
                    continue

                lower_bound_iqr = Q1 - 1.5 * IQR
                upper_bound_iqr = Q3 + 1.5 * IQR

                print(f"1.5*IQR Lower Bound: {lower_bound_iqr:.2f}")
                print(f"1.5*IQR Upper Bound: {upper_bound_iqr:.2f}")

                column_outlier_indices = df_cleaned[
                    (df_cleaned[column] < lower_bound_iqr) | (df_cleaned[column] > upper_bound_iqr)
                ].index
                print(f"Found {len(column_outlier_indices)} potential outliers based on 1.5*IQR.")
                all_potential_outlier_indices.update(column_outlier_indices)

            except Exception as e:
                print(f"An error occurred during IQR calculation for {column}: {e}")

        print(f"\nTotal unique rows flagged as potential outliers by 1.5*IQR: {len(all_potential_outlier_indices)}")

        # --- Step 2.2 & 2.3: Apply Strict Removal Criteria ---
        print("\n--- Step 2.2 & 2.3: Applying Strict Removal Criteria to Potential Outliers ---")
        if not all_potential_outlier_indices:
            print("No potential outliers were identified. No rows to check for removal.")
        else:
            print(f"Checking {len(all_potential_outlier_indices)} potential outlier rows against strict criteria...")
            for idx in all_potential_outlier_indices:
                try:
                    row = df_cleaned.loc[idx]
                    remove_flag = False

                    # Apply rules for rating columns
                    if 'potential' in OUTLIER_COLUMNS_TO_CHECK and 'potential' in row:
                        if pd.notna(row['potential']) and (row['potential'] < RATING_LOWER_BOUND or row['potential'] > RATING_UPPER_BOUND):
                            # print(f"  - Flagged for removal (Index: {idx}): Potential ({row['potential']}) out of range ({RATING_LOWER_BOUND}-{RATING_UPPER_BOUND}).")
                            remove_flag = True
                    if not remove_flag and 'overall_rating' in OUTLIER_COLUMNS_TO_CHECK and 'overall_rating' in row:
                        if pd.notna(row['overall_rating']) and (row['overall_rating'] < RATING_LOWER_BOUND or row['overall_rating'] > RATING_UPPER_BOUND):
                            # print(f"  - Flagged for removal (Index: {idx}): Overall Rating ({row['overall_rating']}) out of range ({RATING_LOWER_BOUND}-{RATING_UPPER_BOUND}).")
                            remove_flag = True

                    # Apply rules for financial columns (check for < 0)
                    if not remove_flag and 'wage_euro' in OUTLIER_COLUMNS_TO_CHECK and 'wage_euro' in row:
                        if pd.notna(row['wage_euro']) and row['wage_euro'] < FINANCIAL_LOWER_BOUND:
                            # print(f"  - Flagged for removal (Index: {idx}): Wage Euro ({row['wage_euro']}) is negative.")
                            remove_flag = True
                    if not remove_flag and 'value_euro' in OUTLIER_COLUMNS_TO_CHECK and 'value_euro' in row:
                         if pd.notna(row['value_euro']) and row['value_euro'] < FINANCIAL_LOWER_BOUND:
                            # print(f"  - Flagged for removal (Index: {idx}): Value Euro ({row['value_euro']}) is negative.")
                            remove_flag = True
                    if not remove_flag and 'release_clause_euro' in OUTLIER_COLUMNS_TO_CHECK and 'release_clause_euro' in row:
                        if pd.notna(row['release_clause_euro']) and row['release_clause_euro'] < FINANCIAL_LOWER_BOUND:
                            # print(f"  - Flagged for removal (Index: {idx}): Release Clause ({row['release_clause_euro']}) is negative.")
                            remove_flag = True

                    if remove_flag:
                        indices_to_remove_outliers.add(idx)

                except KeyError as e:
                    # This might happen if a column used in checks was dropped earlier
                    print(f"Warning: Could not access data for index {idx}, possibly due to missing column {e}. Skipping check for this index.")
                    continue
                except Exception as e:
                     print(f"Warning: An unexpected error occurred checking index {idx}: {e}. Skipping.")
                     continue


            print(f"\nBased on strict criteria, {len(indices_to_remove_outliers)} rows will be removed.")
            # if len(indices_to_remove_outliers) > 0:
            #     print("Indices marked for removal:", sorted(list(indices_to_remove_outliers)))


        # --- Step 2.4: Remove Marked Rows ---
        print("\n--- Step 2.4: Removing Marked Outlier Rows ---")
        if not indices_to_remove_outliers:
            print("No rows met the strict criteria for outlier removal.")
        else:
            print(f"Removing {len(indices_to_remove_outliers)} rows...")
            rows_to_remove_list = sorted(list(indices_to_remove_outliers))
            initial_rows_before_outlier_drop = df_cleaned.shape[0]
            # Ensure indices exist before attempting drop (might have been dropped by NaN removal)
            rows_to_remove_list = [idx for idx in rows_to_remove_list if idx in df_cleaned.index]
            df_cleaned = df_cleaned.drop(index=rows_to_remove_list)
            print(f"Rows removed successfully. {len(rows_to_remove_list)} rows dropped in this step.")

        # --- Verification (Optional) ---
        print("\n--- Verification: Checking remaining min/max for key columns ---")
        for col in OUTLIER_COLUMNS_TO_CHECK:
            if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]) and not df_cleaned[col].isnull().all():
                 # Check if column is not all NaN before calling min/max
                print(f"{col}: Min = {df_cleaned[col].min()}, Max = {df_cleaned[col].max()}")
            elif col not in df_cleaned.columns:
                print(f"{col}: Column not present.")
            elif not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                 print(f"{col}: Not numeric.")
            else: # Handle case where column exists, is numeric, but all values are NaN
                 print(f"{col}: All values are NaN.")

else:
    print("\n--- [SKIPPING] 2. Handling Outliers ---")


# =============================================================================
# 3. Feature Analysis: Correlation
# =============================================================================
if RUN_SECTION_3_CORRELATION:
    print("\n--- [EXECUTING] 3. Feature Analysis: Correlation ---")

    if df_cleaned.empty:
        print("DataFrame is empty. Skipping correlation analysis.")
    else:
        print(f"\nInput DataFrame shape for correlation: {df_cleaned.shape}")
        total_rows_corr_input = len(df_cleaned)
        valid_target_cols_corr = []
        potentially_irrelevant_features_corr = []
        features_to_drop_corr = []

        # --- Step 3.1: Clean and Verify Target Columns ---
        print("\n--- Step 3.1: Verifying target columns and data types ---")
        for col in CORRELATION_TARGET_COLUMNS:
            if col not in df_cleaned.columns:
                print(f" - Warning: Target '{col}' not found.")
                continue

            # Attempt to clean common currency/value formats if object type
            if df_cleaned[col].dtype == 'object':
                print(f" - Attempting object column cleaning for '{col}'...")
                try:
                    # Create a temporary series for cleaning
                    cleaned_series = df_cleaned[col].astype(str).str.replace('€', '', regex=False)
                    # Handle M and K suffixes for millions and thousands
                    cleaned_series = cleaned_series.str.replace(r'M$', 'e6', regex=True)
                    cleaned_series = cleaned_series.str.replace(r'K$', 'e3', regex=True)
                    # Convert to numeric, coercing errors to NaN
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    # Only assign back if conversion was somewhat successful (check not all NaN)
                    if not numeric_series.isnull().all():
                         df_cleaned.loc[:, col] = numeric_series
                         print(f"   - Successfully converted '{col}' to numeric.")
                    else:
                         print(f"   - Warning: Conversion of '{col}' resulted in all NaNs. Skipping.")
                         continue
                except Exception as e:
                    print(f"   - Warning: Failed to clean/convert '{col}': {e}. Skipping.")
                    continue

            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                print(f" - Warning: Target '{col}' not numeric (Type: {df_cleaned[col].dtype}). Skipping.")
                continue

            nan_count = df_cleaned[col].isna().sum()
            nan_percentage = (nan_count / total_rows_corr_input) * 100 if total_rows_corr_input > 0 else 0
            print(f" - Target '{col}' | Type: {df_cleaned[col].dtype} | Missing: {nan_count} ({nan_percentage:.2f}%)")

            if nan_percentage > MAX_MISSING_TARGET_PERCENTAGE:
                print(f"   -> Excluding '{col}' due to > {MAX_MISSING_TARGET_PERCENTAGE}% missing.")
            else:
                valid_target_cols_corr.append(col)

        if not valid_target_cols_corr:
            print("\nError: No valid numeric target columns available for correlation analysis. Skipping.")
        else:
            print(f"\nUsing valid target columns for correlation: {valid_target_cols_corr}")

            # --- Step 3.2: Identify Predictor Columns ---
            numerical_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
            predictor_cols_corr = [col for col in numerical_cols if col not in valid_target_cols_corr]
            cols_for_corr_matrix = predictor_cols_corr + valid_target_cols_corr

            if not predictor_cols_corr:
                print("\nWarning: No numerical predictor columns found. Correlation analysis might be limited.")
            else:
                # --- Step 3.3: Calculate Correlation (on complete rows) ---
                print("\n--- Step 3.3: Calculating Correlation Matrix ---")
                # Use only rows with complete data for the selected columns for correlation
                df_corr_subset = df_cleaned[cols_for_corr_matrix].dropna()
                print(f"Using {len(df_corr_subset)} rows with complete data for correlation calculation.")

                if len(df_corr_subset) < 2:
                     print("Warning: Not enough complete rows (< 2) to calculate correlation.")
                     corr_matrix = pd.DataFrame() # Empty dataframe
                else:
                    try:
                        corr_matrix = df_corr_subset.corr()
                        if corr_matrix.empty or corr_matrix.isnull().values.any():
                            print("Warning: Correlation matrix is empty or contains NaNs after calculation.")
                            corr_matrix = pd.DataFrame() # Reset if problematic

                    except Exception as e:
                        print(f"Error calculating correlation matrix: {e}")
                        corr_matrix = pd.DataFrame() # Reset on error

                # --- Step 3.4: Identify Irrelevant Features ---
                if not corr_matrix.empty:
                    print(f"\n--- Step 3.4: Identifying Features with |Correlation| < {CORRELATION_THRESHOLD} to ALL Targets ---")
                    # Ensure target columns actually exist in the calculated matrix
                    targets_in_matrix = [t for t in valid_target_cols_corr if t in corr_matrix.columns]
                    predictors_in_matrix = [p for p in predictor_cols_corr if p in corr_matrix.index]

                    if not targets_in_matrix or not predictors_in_matrix:
                         print("Warning: Could not find valid targets or predictors in the correlation matrix index/columns.")
                    else:
                        target_correlations = corr_matrix.loc[predictors_in_matrix, targets_in_matrix]

                        for feature in predictors_in_matrix:
                            try:
                                # Ensure the feature row exists before accessing
                                if feature in target_correlations.index:
                                    abs_corr_values = target_correlations.loc[feature, targets_in_matrix].abs()
                                    if not abs_corr_values.empty and not abs_corr_values.isna().any():
                                        if (abs_corr_values < CORRELATION_THRESHOLD).all():
                                            potentially_irrelevant_features_corr.append(feature)
                                    # else: handle cases where correlations might be NaN if needed
                                else:
                                    print(f"Warning: Feature '{feature}' unexpectedly not found in target correlation index. Skipping.")

                            except KeyError:
                                # This could happen if a target column was missing, already handled above but for safety
                                print(f"Warning: Target column lookup failed for feature '{feature}'. Skipping.")
                                continue
                            except Exception as e:
                                print(f"Warning: Unexpected error checking correlation for feature '{feature}': {e}. Skipping.")


                        if potentially_irrelevant_features_corr:
                            print(f"Found {len(potentially_irrelevant_features_corr)} features potentially irrelevant to all targets:")
                            #for i, feature in enumerate(potentially_irrelevant_features_corr): print(f"  {i+1}. {feature}")
                            print(f"   {potentially_irrelevant_features_corr}")
                            features_to_drop_corr = potentially_irrelevant_features_corr
                        else:
                            print("No features identified with low correlation to all targets.")

                # --- Step 3.5: Drop Irrelevant Features ---
                print("\n--- Step 3.5: Dropping Low-Correlation Features ---")
                if features_to_drop_corr:
                    print(f"Proceeding to automatically drop {len(features_to_drop_corr)} features...")
                    print("WARNING: This drops features based ONLY on weak LINEAR correlation.")

                    try:
                        features_to_drop_existing = [f for f in features_to_drop_corr if f in df_cleaned.columns]
                        if len(features_to_drop_existing) < len(features_to_drop_corr):
                            print("Note: Some identified features might have already been missing.")

                        df_cleaned = df_cleaned.drop(columns=features_to_drop_existing)
                        print(f"Features dropped successfully. New DataFrame shape: {df_cleaned.shape}")

                    except Exception as e:
                        print(f"Error during column drop: {e}")
                else:
                    print("No features were identified for dropping based on correlation.")

                # --- Step 3.6: Optional Final Heatmap ---
                print("\n--- Step 3.6: Optional Final Heatmap ---")
                if not df_cleaned.empty:
                     numerical_cols_after_drop = df_cleaned.select_dtypes(include=np.number).columns.tolist()
                     if len(numerical_cols_after_drop) > 1:
                         df_final_corr_subset = df_cleaned[numerical_cols_after_drop].dropna()
                         print(f"Using {len(df_final_corr_subset)} complete rows for final heatmap.")
                         if len(df_final_corr_subset) >= 2:
                             try:
                                 final_corr_matrix = df_final_corr_subset.corr()
                                 if not final_corr_matrix.empty:
                                     fig_width = max(10, len(final_corr_matrix.columns) * 0.6)
                                     fig_height = max(8, len(final_corr_matrix.columns) * 0.5)
                                     plt.figure(figsize=(fig_width, fig_height))
                                     sns.heatmap(final_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
                                     plt.title(f'Final Correlation Matrix (After Cleaning/Dropping)')
                                     plt.xticks(rotation=60, ha='right')
                                     plt.yticks(rotation=0)
                                     plt.tight_layout()
                                     plt.show()
                                 else: print("Could not compute final correlation matrix.")
                             except Exception as e: print(f"Could not display final heatmap: {e}")
                         else: print("Skipping final heatmap: Not enough complete rows (< 2).")
                     else: print("Skipping final heatmap: Not enough numerical columns remaining.")
                else: print("Skipping final heatmap: DataFrame is empty.")

else:
    print("\n--- [SKIPPING] 3. Feature Analysis: Correlation ---")


# =============================================================================
# 4. Feature Analysis: ANOVA (Nationality vs Target)
# =============================================================================
if RUN_SECTION_4_ANOVA:
    print(f"\n--- [EXECUTING] 4. Feature Analysis: ANOVA ({TARGET_FOR_ANOVA} by Nationality) ---")

    if df_cleaned.empty:
        print("DataFrame is empty. Skipping ANOVA.")
    elif 'nationality' not in df_cleaned.columns:
        print("Error: 'nationality' column not found. Skipping ANOVA.")
    elif TARGET_FOR_ANOVA not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[TARGET_FOR_ANOVA]):
        print(f"Error: Target '{TARGET_FOR_ANOVA}' not found or not numeric. Skipping ANOVA.")
    else:
        # --- Step 4.1: Group Nationalities ---
        print(f"\n--- Step 4.1: Grouping nationalities (Top {TOP_N_NATIONALITIES} and 'Other') ---")
        temp_group_col_name = 'nation_group_temp'
        if temp_group_col_name in df_cleaned.columns: # Avoid creating if it somehow exists
            print(f"Warning: Temporary column '{temp_group_col_name}' already exists. Attempting to overwrite.")
            df_cleaned.drop(columns=[temp_group_col_name], inplace=True, errors='ignore')

        try:
            nationality_counts = df_cleaned['nationality'].value_counts()
            # Ensure we don't request more top N than unique nationalities available
            actual_top_n = min(TOP_N_NATIONALITIES, len(nationality_counts))
            if actual_top_n < TOP_N_NATIONALITIES:
                print(f"  Note: Requested top {TOP_N_NATIONALITIES}, but only {actual_top_n} unique nationalities exist.")

            top_nationalities = nationality_counts.nlargest(actual_top_n).index.tolist()
            df_cleaned[temp_group_col_name] = df_cleaned['nationality'].apply(lambda x: x if x in top_nationalities else 'Other')
            print("Top nationality groups considered:", top_nationalities)
            anova_step1_success = True
        except Exception as e:
             print(f"Error grouping nationalities: {e}. Skipping ANOVA.")
             anova_step1_success = False
             # Attempt cleanup if column creation failed partially
             if temp_group_col_name in df_cleaned.columns: df_cleaned.drop(columns=[temp_group_col_name], inplace=True, errors='ignore')


        if anova_step1_success: # Proceed only if grouping succeeded
            # --- Step 4.2: Prepare Data ---
            print(f"\n--- Step 4.2: Preparing data for ANOVA ---")
            anova_data = df_cleaned[[temp_group_col_name, TARGET_FOR_ANOVA]].dropna()

            if anova_data.empty:
                print("Error: No data remaining after dropping NaNs for ANOVA.")
            else:
                print(f"Using {len(anova_data)} rows for ANOVA.")
                unique_groups_anova = anova_data[temp_group_col_name].unique()
                print("Unique nationality groups in ANOVA data:", unique_groups_anova)

                # --- Step 4.3: Perform ANOVA ---
                if len(unique_groups_anova) < 2:
                    print("Error: Need at least two groups with data for ANOVA.")
                else:
                    print("\n--- Step 4.3: Performing ANOVA test ---")
                    grouped_data_anova = [anova_data[TARGET_FOR_ANOVA][anova_data[temp_group_col_name] == group] for group in unique_groups_anova]
                    try:
                        f_statistic, p_value = stats.f_oneway(*grouped_data_anova)
                        print("\nSciPy ANOVA Results:")
                        print(f" - F-statistic: {f_statistic:.4f}")
                        print(f" - P-value:     {p_value:.4e}") # Scientific notation

                        # --- Step 4.4: Interpret Results ---
                        print(f"\nInterpretation (Alpha = {ALPHA_LEVEL}):")
                        if p_value > ALPHA_LEVEL:
                            print(f" - Conclusion: Fail to reject null hypothesis (p > {ALPHA_LEVEL}).")
                            print(f" - Suggests NO strong evidence that mean '{TARGET_FOR_ANOVA}' differs significantly across nationality groups.")
                            print(" - ANOVA supports considering 'Nationality' potentially less important based on mean difference.")
                            anova_decision = "DROP / LESS IMPORTANT"
                        else:
                            print(f" - Conclusion: Reject null hypothesis (p <= {ALPHA_LEVEL}).")
                            print(f" - Suggests strong evidence that mean '{TARGET_FOR_ANOVA}' DOES differ across some nationality groups.")
                            print(" - ANOVA supports KEEPING 'Nationality' as potentially important.")
                            anova_decision = "KEEP / IMPORTANT"
                        print(f"ANOVA evidence supports potentially: {anova_decision}")

                    except Exception as e:
                        print(f"Error during ANOVA calculation: {e}")

            # --- Step 4.5: Optional Boxplot ---
            print("\n--- Step 4.5: Optional Boxplot ---")
            # Check if anova_data was created and has groups
            if 'anova_data' in locals() and not anova_data.empty and 'unique_groups_anova' in locals() and len(unique_groups_anova) >= 1:
                 try:
                     plt.figure(figsize=(max(12, len(unique_groups_anova) * 0.7), 6))
                     # Define order to put 'Other' last if it exists
                     plot_order = sorted([g for g in unique_groups_anova if g != 'Other'])
                     if 'Other' in unique_groups_anova: plot_order.append('Other')

                     sns.boxplot(data=anova_data, x=temp_group_col_name, y=TARGET_FOR_ANOVA, order=plot_order)
                     plt.title(f'Boxplot of {TARGET_FOR_ANOVA} by Nationality Group')
                     plt.xticks(rotation=75, ha='right')
                     plt.tight_layout()
                     plt.show()
                 except Exception as e: print(f"Could not display boxplot: {e}")
            else: print("Skipping boxplot: Not enough data or groups after NaN drop.")


            # --- Step 4.6: Cleanup ---
            if temp_group_col_name in df_cleaned.columns:
                df_cleaned.drop(columns=[temp_group_col_name], inplace=True, errors='ignore')
                print(f"\nTemporary '{temp_group_col_name}' column removed.")

else:
    print("\n--- [SKIPPING] 4. Feature Analysis: ANOVA ---")


# =============================================================================
# 5. Feature Analysis: RandomForest Importance (Nationality vs Core)
# =============================================================================
if RUN_SECTION_5_RANDOM_FOREST:
    print(f"\n--- [EXECUTING] 5. Feature Analysis: RandomForest Importance ('{CATEGORICAL_FEATURE_RF}' vs Core) ---")
    print(f"Target Variable: {TARGET_FOR_RF}")

    # encoded_feature_name_rf is defined in config section

    # --- Pre-checks for RF ---
    rf_possible = True
    if df_cleaned.empty:
        print("DataFrame is empty. Skipping RandomForest analysis.")
        rf_possible = False
    elif TARGET_FOR_RF not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[TARGET_FOR_RF]):
        print(f"Error: Target '{TARGET_FOR_RF}' not found or not numeric. Skipping RF.")
        rf_possible = False
    elif CATEGORICAL_FEATURE_RF not in df_cleaned.columns:
         print(f"Error: Categorical feature '{CATEGORICAL_FEATURE_RF}' not found. Skipping RF.")
         rf_possible = False
    else: # Check core predictors
        missing_core = []
        non_numeric_core = []
        for pred in CORE_PREDICTORS_RF:
            if pred not in df_cleaned.columns: missing_core.append(pred)
            elif not pd.api.types.is_numeric_dtype(df_cleaned[pred]): non_numeric_core.append(pred)
        if missing_core: print(f"Error: Core predictors not found: {missing_core}. Skipping RF."); rf_possible = False
        if non_numeric_core: print(f"Error: Core predictors not numeric: {non_numeric_core}. Skipping RF."); rf_possible = False

    if rf_possible:
        # --- Step 5.1: Frequency Encode ---
        print(f"\n--- Step 5.1: Frequency Encoding '{CATEGORICAL_FEATURE_RF}' ---")
        try:
            if encoded_feature_name_rf in df_cleaned.columns:
                 print(f"Warning: Encoded column '{encoded_feature_name_rf}' already exists. Overwriting.")
            nationality_freq = df_cleaned[CATEGORICAL_FEATURE_RF].value_counts(normalize=True)
            df_cleaned[encoded_feature_name_rf] = df_cleaned[CATEGORICAL_FEATURE_RF].map(nationality_freq)
            # Fill NaNs that might result from original NaNs in categorical column or unseen values if split first
            df_cleaned[encoded_feature_name_rf].fillna(0, inplace=True)
            print(f"Created '{encoded_feature_name_rf}' column.")
        except Exception as e:
             print(f"Error during frequency encoding: {e}. Skipping RF analysis.")
             rf_possible = False # Stop RF if encoding fails
             # Cleanup potentially partially created column
             if encoded_feature_name_rf in df_cleaned.columns: df_cleaned.drop(columns=[encoded_feature_name_rf], inplace=True, errors='ignore')

    if rf_possible:
        # --- Step 5.2: Define Features & Target, Handle NaNs ---
        print("\n--- Step 5.2: Defining Features (X) and Target (y), Handling NaNs ---")
        features_for_model_rf = CORE_PREDICTORS_RF + [encoded_feature_name_rf]
        print("Features included in the model:", features_for_model_rf)

        # Select data and handle NaNs specific to this model run
        X_rf = df_cleaned[features_for_model_rf].copy()
        y_rf = df_cleaned[TARGET_FOR_RF].copy()

        # Drop rows where *any* selected feature OR the target is NaN for this model
        combined_rf = pd.concat([X_rf, y_rf], axis=1)
        initial_rows_rf = len(combined_rf)
        combined_rf.dropna(inplace=True)
        rows_after_na_drop_rf = len(combined_rf)

        if combined_rf.empty:
             print("Error: No data remaining after dropping NaNs for RF model. Skipping.")
             rf_possible = False # Stop RF
        else:
            X_rf = combined_rf[features_for_model_rf]
            y_rf = combined_rf[TARGET_FOR_RF]
            print(f"Using {rows_after_na_drop_rf} rows for model (dropped {initial_rows_rf - rows_after_na_drop_rf} rows with NaNs).")

    if rf_possible:
        # --- Step 5.3: Train/Test Split ---
        print("\n--- Step 5.3: Splitting data ---")
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X_rf, y_rf, test_size=0.2, random_state=RF_RANDOM_STATE
        )
        print(f"Training set size: {len(X_train_rf)}, Testing set size: {len(X_test_rf)}")

        # --- Step 5.4: Train RandomForest ---
        print("\n--- Step 5.4: Training RandomForestRegressor ---")
        model_rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            n_jobs=RF_N_JOBS,
            oob_score=False # Can set to True for OOB R^2, but slower
        )
        try:
            model_rf.fit(X_train_rf, y_train_rf)
            print("Model training complete.")
        except Exception as e:
             print(f"Error during model training: {e}. Skipping importance analysis.")
             rf_possible = False # Stop RF

    if rf_possible:
        # --- Step 5.5: Feature Importances ---
        print("\n--- Step 5.5: Extracting Feature Importances ---")
        try:
            importances_rf = model_rf.feature_importances_
            feature_names_rf = X_rf.columns
            importance_df_rf = pd.DataFrame({
                'Feature': feature_names_rf,
                'Importance': importances_rf
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

            print(importance_df_rf)

            # Plot importances
            plt.figure(figsize=(10, max(4, len(importance_df_rf) * 0.5)))
            sns.barplot(x='Importance', y='Feature', data=importance_df_rf, palette='viridis')
            plt.title('Feature Importances from RandomForest')
            plt.xlabel('Importance Score (Gini Importance)')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

            # --- Step 5.6: Analyze and Recommend ---
            print("\n--- Step 5.6: Analyzing Importance and Recommendation ---")
            # Ensure the encoded feature exists in the importance results before accessing
            if encoded_feature_name_rf in importance_df_rf['Feature'].values:
                nationality_importance_rf = importance_df_rf.loc[importance_df_rf['Feature'] == encoded_feature_name_rf, 'Importance'].iloc[0]
                nationality_rank_rf = importance_df_rf[importance_df_rf['Feature'] == encoded_feature_name_rf].index[0] + 1
                print(f"Importance score for '{encoded_feature_name_rf}': {nationality_importance_rf:.4f} (Rank: {nationality_rank_rf}/{len(importance_df_rf)})")

                # Compare to weakest core predictor
                core_importances = importance_df_rf[importance_df_rf['Feature'].isin(CORE_PREDICTORS_RF)]
                if not core_importances.empty:
                    min_core_importance_rf = core_importances['Importance'].min()
                    min_core_feature_rf = core_importances.loc[core_importances['Importance'] == min_core_importance_rf, 'Feature'].iloc[0]
                    print(f"Compare to weakest core predictor '{min_core_feature_rf}': {min_core_importance_rf:.4f}")

                    rf_decision = "KEEP" # Default
                    # Example criteria for dropping: Low absolute importance AND significantly less than weakest core
                    if nationality_importance_rf < 0.01 and nationality_importance_rf < min_core_importance_rf * 0.5:
                        print(f"\nRecommendation: DROP '{CATEGORICAL_FEATURE_RF}'.")
                        print(f"Reason: Importance ({nationality_importance_rf:.4f}) is very low (< 0.01) and < 50% of weakest core predictor ('{min_core_feature_rf}' at {min_core_importance_rf:.4f}).")
                        rf_decision = "DROP"
                    elif nationality_importance_rf < min_core_importance_rf:
                         print(f"\nRecommendation: CONSIDER DROPPING '{CATEGORICAL_FEATURE_RF}'.")
                         print(f"Reason: Importance ({nationality_importance_rf:.4f}) is lower than weakest core predictor ('{min_core_feature_rf}' at {min_core_importance_rf:.4f}).")
                         rf_decision = "CONSIDER DROPPING"
                    else:
                        print(f"\nRecommendation: KEEP '{CATEGORICAL_FEATURE_RF}'.")
                        print(f"Reason: Importance ({nationality_importance_rf:.4f}) >= weakest core predictor ('{min_core_feature_rf}' at {min_core_importance_rf:.4f}).")
                        rf_decision = "KEEP"
                    print(f"RF evidence supports potentially: {rf_decision}")
                else:
                     print("Could not find core predictor importances for comparison.")
            else:
                 print(f"Warning: Encoded feature '{encoded_feature_name_rf}' not found in importance results. Cannot provide recommendation.")

        except Exception as e:
             print(f"Error during importance extraction or analysis: {e}")


    # --- Step 5.7: Optional Cleanup of Encoded Feature ---
    # Decide whether to keep the frequency encoded column based on analysis.
    # Currently, it remains in df_cleaned unless manually removed.
    if encoded_feature_name_rf in df_cleaned.columns:
        print(f"\nNote: Frequency encoded column '{encoded_feature_name_rf}' currently remains in df_cleaned.")
        # Example: if rf_decision == "DROP":
        #     print(f"Removing '{encoded_feature_name_rf}' as per recommendation.")
        #     df_cleaned.drop(columns=[encoded_feature_name_rf], inplace=True, errors='ignore')

else:
    print("\n--- [SKIPPING] 5. Feature Analysis: RandomForest Importance ---")
    # Ensure the variable exists even if skipped, for Section 7 check
    if 'encoded_feature_name_rf' not in locals():
         encoded_feature_name_rf = f'{CATEGORICAL_FEATURE_RF}_freq_encoded'


# =============================================================================
# 7. Feature Analysis: Linear Regression (Example)
# =============================================================================
if RUN_SECTION_7_LINEAR_REGRESSION:
    print(f"\n--- [EXECUTING] 7. Feature Analysis: Linear Regression ({TARGET_FOR_RF} vs Core) ---")

    # Use the same target and core predictors as RF for comparison
    # TARGET_FOR_RF = 'value_euro'
    # CORE_PREDICTORS_RF = ['overall_rating', 'potential', 'age']
    # Reuse the frequency encoded feature if it exists and wasn't dropped
    lr_features = CORE_PREDICTORS_RF.copy()
    if encoded_feature_name_rf in df_cleaned.columns:
         lr_features.append(encoded_feature_name_rf)
         print(f"Including '{encoded_feature_name_rf}' in Linear Regression features.")
    else:
         print(f"'{encoded_feature_name_rf}' not found or previously dropped, using only core predictors for LR.")

    # --- Pre-checks for LR ---
    lr_possible = True
    if df_cleaned.empty:
        print("DataFrame is empty. Skipping Linear Regression.")
        lr_possible = False
    elif TARGET_FOR_RF not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[TARGET_FOR_RF]):
        print(f"Error: Target '{TARGET_FOR_RF}' not found or not numeric. Skipping LR.")
        lr_possible = False
    else:
        missing_lr_features = [f for f in lr_features if f not in df_cleaned.columns]
        non_numeric_lr = [f for f in lr_features if f in df_cleaned.columns and not pd.api.types.is_numeric_dtype(df_cleaned[f])]
        if missing_lr_features: print(f"Error: LR predictors not found: {missing_lr_features}. Skipping LR."); lr_possible = False
        if non_numeric_lr: print(f"Error: LR predictors not numeric: {non_numeric_lr}. Skipping LR."); lr_possible = False

    if lr_possible:
        # --- Step 7.1: Prepare Data (Select columns, drop NaNs specifically for LR) ---
        print("\n--- Step 7.1: Preparing data for Linear Regression ---")
        X_lr_raw = df_cleaned[lr_features].copy()
        y_lr = df_cleaned[TARGET_FOR_RF].copy()

        combined_lr = pd.concat([X_lr_raw, y_lr], axis=1)
        initial_rows_lr = len(combined_lr)
        combined_lr.dropna(inplace=True)
        rows_after_na_drop_lr = len(combined_lr)

        if combined_lr.empty:
            print("Error: No data remaining after dropping NaNs for LR model. Skipping.")
            lr_possible = False
        else:
            X_lr_raw = combined_lr[lr_features]
            y_lr = combined_lr[TARGET_FOR_RF]
            print(f"Using {rows_after_na_drop_lr} rows for LR model (dropped {initial_rows_lr - rows_after_na_drop_lr} rows with NaNs).")

    if lr_possible:
        # --- Step 7.2: Train/Test Split ---
        print("\n--- Step 7.2: Splitting data ---")
        X_train_lr_raw, X_test_lr_raw, y_train_lr, y_test_lr = train_test_split(
            X_lr_raw, y_lr, test_size=0.2, random_state=RF_RANDOM_STATE # Use same random state
        )
        print(f"Training set size: {len(X_train_lr_raw)}, Testing set size: {len(X_test_lr_raw)}")

        # --- Step 7.3: Feature Scaling (Important for LR, especially if comparing coefficients) ---
        print("\n--- Step 7.3: Scaling features using StandardScaler ---")
        scaler = StandardScaler()
        # Fit only on training data, transform both train and test
        X_train_lr_scaled = scaler.fit_transform(X_train_lr_raw)
        X_test_lr_scaled = scaler.transform(X_test_lr_raw)

        # Optional: Convert scaled arrays back to DataFrames for easier inspection if needed
        # X_train_lr_scaled_df = pd.DataFrame(X_train_lr_scaled, columns=lr_features, index=X_train_lr_raw.index)
        # X_test_lr_scaled_df = pd.DataFrame(X_test_lr_scaled, columns=lr_features, index=X_test_lr_raw.index)


        # --- Step 7.4: Train Linear Regression Model ---
        print("\n--- Step 7.4: Training Linear Regression model ---")
        model_lr = LinearRegression()
        try:
            model_lr.fit(X_train_lr_scaled, y_train_lr)
            print("Linear Regression model training complete.")

            # --- Step 7.5: Make Predictions & Evaluate Model ---
            print("\n--- Step 7.5: Evaluating Linear Regression model ---")
            y_pred_lr = model_lr.predict(X_test_lr_scaled)

            # Calculate metrics
            r2_lr = r2_score(y_test_lr, y_pred_lr)
            mae_lr = mean_absolute_error(y_test_lr, y_pred_lr)
            mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
            rmse_lr = np.sqrt(mse_lr)

            print(f"  R-squared (R²): {r2_lr:.4f}")
            print(f"  Mean Absolute Error (MAE): {mae_lr:,.2f}")
            print(f"  Mean Squared Error (MSE): {mse_lr:,.2f}")
            print(f"  Root Mean Squared Error (RMSE): {rmse_lr:,.2f}")

            # --- Step 7.6: Inspect Coefficients (Optional) ---
            print("\n--- Step 7.6: Model Coefficients ---")
            # Coefficients represent the change in the target for a one-unit change
            # in the predictor, *assuming the predictor was scaled*.
            coeffs = pd.DataFrame({
                'Feature': lr_features,
                'Coefficient': model_lr.coef_
            }).sort_values(by='Coefficient', ascending=False)
            print(coeffs)
            print(f"\nIntercept: {model_lr.intercept_:,.2f}")

            # --- Step 7.7: Visualizing Results ---
            print("\n--- Step 7.7: Visualizing Linear Regression Results ---")

            # 1. Actual vs. Predicted Plot
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test_lr, y_pred_lr, alpha=0.5, edgecolor='k', s=50)
            # Add line of perfect fit (y=x)
            min_val = min(y_test_lr.min(), y_pred_lr.min())
            max_val = max(y_test_lr.max(), y_pred_lr.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y=x)')
            plt.title(f'Actual vs. Predicted Player Values (LR)\nR² = {r2_lr:.4f}', fontsize=14)
            plt.xlabel('Actual Value (€)', fontsize=12)
            plt.ylabel('Predicted Value (€)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # 2. Residuals vs. Predicted Plot
            residuals_lr = y_test_lr - y_pred_lr
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred_lr, residuals_lr, alpha=0.5, edgecolor='k', s=50)
            # Add horizontal line at zero residual
            plt.axhline(0, color='red', linestyle='--', lw=2, label='Zero Residual')
            plt.title('Residuals vs. Predicted Values (LR)', fontsize=14)
            plt.xlabel('Predicted Value (€)', fontsize=12)
            plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            print("Residual Plot Analysis: Look for random scatter around y=0.")
            print("Patterns (like curves or funnels) indicate violations of LR assumptions (non-linearity, heteroscedasticity).")


            # 3. Residual Distribution Plot
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals_lr, kde=True, bins=30, color='skyblue', edgecolor='black')
            plt.title('Distribution of Residuals (LR)', fontsize=14)
            plt.xlabel('Residual Value (€)', fontsize=12)
            plt.ylabel('Frequency / Density', fontsize=12)
            plt.axvline(residuals_lr.mean(), color='red', linestyle='--', lw=1.5, label=f'Mean Residual: {residuals_lr.mean():.2f}')
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            print("Residual Distribution Analysis: Check if distribution is roughly normal (bell-shaped) and centered near zero.")


        except Exception as e:
             print(f"Error during Linear Regression training, evaluation or visualization: {e}")

else:
    print("\n--- [SKIPPING] 7. Feature Analysis: Linear Regression ---")


# =============================================================================
# 8. Advanced Modeling: LightGBM for Value Prediction
# =============================================================================

# --- !! Add this flag at the top in the Control Section !! ---
# RUN_SECTION_8_LIGHTGBM = True # Set to True to run LightGBM Model

if RUN_SECTION_8_LIGHTGBM:
    print("\n--- [EXECUTING] 8. Advanced Modeling: LightGBM for Value Prediction ---")

    # --- Imports for this section ---
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        print("Required libraries for LightGBM loaded.")
    except ImportError as e:
        print(f"Error importing libraries for LightGBM: {e}")
        print("Please install lightgbm: pip install lightgbm")
        # Skip the rest of this section if imports fail
        RUN_SECTION_8_LIGHTGBM = False


if RUN_SECTION_8_LIGHTGBM: # Check again in case import failed

    if df_cleaned.empty:
        print("DataFrame 'df_cleaned' is empty. Skipping LightGBM modeling.")
    elif TARGET_FOR_RF not in df_cleaned.columns: # Using TARGET_FOR_RF ('value_euro')
         print(f"Target column '{TARGET_FOR_RF}' not found. Skipping LightGBM modeling.")
    else:
        # --- Step 8.1: Define Features and Target ---
        print(f"\n--- Step 8.1: Defining Features and Target ({TARGET_FOR_RF}) ---")

        # Select a broader set of features potentially relevant to value
        # Adjust this list based on EDA, domain knowledge, and previous analyses
        lgbm_features_numeric = [
            'overall_rating', 'potential', 'age',
            'wage_euro', 'release_clause_euro', # Financials can be strong predictors but check for data leakage if predicting future value
            'international_reputation', 'skill_moves', 'weak_foot',
            'height_cm', 'weight_kgs',
            # Add more specific skills if desired, e.g.:
            # 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            # 'attacking_crossing', 'attacking_finishing', ... etc.
        ]
        lgbm_features_categorical = [
            'positions', # Will need special handling
            'preferred_foot',
            'work_rate',
            'body_type',
            'nationality', # Will use frequency encoding or treat as category
        ]

        # Check if frequency encoded nationality exists from Section 5
        if encoded_feature_name_rf in df_cleaned.columns:
             print(f"Using frequency encoded nationality '{encoded_feature_name_rf}'.")
             # Add it to numeric features as it's already encoded
             if encoded_feature_name_rf not in lgbm_features_numeric:
                  lgbm_features_numeric.append(encoded_feature_name_rf)
             # Remove raw nationality from categorical list if encoded version is used
             if 'nationality' in lgbm_features_categorical:
                  lgbm_features_categorical.remove('nationality')
        else:
             print(f"Frequency encoded nationality '{encoded_feature_name_rf}' not found. Will treat 'nationality' as a category.")
             # Keep 'nationality' in the categorical list


        lgbm_features_all = lgbm_features_numeric + lgbm_features_categorical
        target_lgbm = TARGET_FOR_RF # 'value_euro'

        # Ensure all selected features exist in the dataframe
        lgbm_features_exist = [f for f in lgbm_features_all if f in df_cleaned.columns]
        missing_features = [f for f in lgbm_features_all if f not in df_cleaned.columns]
        if missing_features:
             print(f"Warning: The following features were not found and will be excluded: {missing_features}")
        lgbm_features_all = lgbm_features_exist # Use only existing features

        # Separate numeric and categorical lists based on existing features
        lgbm_features_numeric = [f for f in lgbm_features_numeric if f in lgbm_features_all]
        lgbm_features_categorical = [f for f in lgbm_features_categorical if f in lgbm_features_all]

        print(f"Numeric Features used: {lgbm_features_numeric}")
        print(f"Categorical Features used: {lgbm_features_categorical}")

        # --- Step 8.2: Prepare Data (Select, Drop Target NaNs) ---
        print("\n--- Step 8.2: Selecting Data and Handling Target NaNs ---")
        X_lgbm_raw = df_cleaned[lgbm_features_all].copy()
        y_lgbm = df_cleaned[target_lgbm].copy()

        # Drop rows where the TARGET variable is missing - crucial!
        target_nan_mask = y_lgbm.isna()
        if target_nan_mask.any():
            print(f"Dropping {target_nan_mask.sum()} rows due to missing target variable '{target_lgbm}'.")
            X_lgbm_raw = X_lgbm_raw[~target_nan_mask]
            y_lgbm = y_lgbm[~target_nan_mask]

        if X_lgbm_raw.empty:
             print(f"Error: No data remaining after dropping rows with missing target '{target_lgbm}'. Skipping LightGBM.")
        else:
            print(f"Data shape for LightGBM after target NaN drop: {X_lgbm_raw.shape}")

            # --- Step 8.3: Handle 'positions' Column (Example: Take first position) ---
            # More sophisticated handling (like multi-label binarization) is possible but complex.
            positions_col = 'positions' # Define variable for clarity
            if positions_col in X_lgbm_raw.columns:
                 print(f"\n--- Step 8.3: Simplifying '{positions_col}' column (using first position) ---")
                 # Fill NaNs in position before splitting, e.g., with 'Unknown'
                 X_lgbm_raw[positions_col] = X_lgbm_raw[positions_col].fillna('Unknown').astype(str)
                 # Take the first position listed
                 X_lgbm_raw[positions_col] = X_lgbm_raw[positions_col].apply(lambda x: x.split(',')[0])
                 print(f"Example simplified positions: {X_lgbm_raw[positions_col].unique()[:10]}")
            else:
                 print(f"Column '{positions_col}' not found in selected features.")


            # --- Step 8.4: Train/Test Split (Split BEFORE imputation/encoding) ---
            print("\n--- Step 8.4: Splitting Data into Training and Testing sets ---")
            X_train_lgbm_raw, X_test_lgbm_raw, y_train_lgbm, y_test_lgbm = train_test_split(
                X_lgbm_raw, y_lgbm, test_size=0.2, random_state=RF_RANDOM_STATE
            )
            print(f"Training set shape: {X_train_lgbm_raw.shape}, Testing set shape: {X_test_lgbm_raw.shape}")


            # --- Step 8.5: Preprocessing Pipeline (Imputation & Encoding) ---
            print("\n--- Step 8.5: Setting up Preprocessing Pipeline ---")

            # Impute numerical features with the median
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
                # No scaling needed for LightGBM usually
            ])

            # Impute categorical features with a constant value and then OneHotEncode
            # handle_unknown='ignore' prevents errors if test set has categories not seen in train
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for easier feature name handling later
            ])

            # Create the column transformer
            # Note: We use the updated lists of numeric/categorical features that exist
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, lgbm_features_numeric),
                    ('cat', categorical_transformer, lgbm_features_categorical)
                ],
                remainder='passthrough' # Keep any columns not specified (shouldn't be any here)
            )

            # --- Step 8.6: Apply Preprocessing ---
            print("\n--- Step 8.6: Applying Preprocessing ---")
            # Fit the preprocessor on the training data and transform both train and test
            X_train_lgbm_processed = preprocessor.fit_transform(X_train_lgbm_raw)
            X_test_lgbm_processed = preprocessor.transform(X_test_lgbm_raw)
            print("Preprocessing complete.")
            print(f"Processed training data shape: {X_train_lgbm_processed.shape}")
            print(f"Processed testing data shape: {X_test_lgbm_processed.shape}")

            # --- Step 8.7: Train LightGBM Model ---
            print("\n--- Step 8.7: Training LightGBM Regressor ---")
            # Basic LightGBM parameters - can be tuned extensively
            lgbm_model = lgb.LGBMRegressor(
                random_state=RF_RANDOM_STATE,
                n_jobs=RF_N_JOBS # Use available cores
                # objective='regression_l1', # MAE loss, less sensitive to outliers than default L2 (MSE)
                # metric='mae',
                # n_estimators=1000, # More trees usually better, use early_stopping
                # learning_rate=0.05,
                # num_leaves=31, # Default
                # You would typically use validation sets and early stopping for tuning
            )

            try:
                lgbm_model.fit(X_train_lgbm_processed, y_train_lgbm)
                print("LightGBM model training complete.")

                # --- Step 8.8: Evaluate Model ---
                print("\n--- Step 8.8: Evaluating LightGBM model ---")
                y_pred_lgbm = lgbm_model.predict(X_test_lgbm_processed)

                # Calculate metrics
                r2_lgbm = r2_score(y_test_lgbm, y_pred_lgbm)
                mae_lgbm = mean_absolute_error(y_test_lgbm, y_pred_lgbm)
                mse_lgbm = mean_squared_error(y_test_lgbm, y_pred_lgbm)
                rmse_lgbm = np.sqrt(mse_lgbm)

                print(f"  R-squared (R²): {r2_lgbm:.4f}")
                print(f"  Mean Absolute Error (MAE): {mae_lgbm:,.2f}")
                print(f"  Mean Squared Error (MSE): {mse_lgbm:,.2f}")
                print(f"  Root Mean Squared Error (RMSE): {rmse_lgbm:,.2f}")
                print("\nCompare these metrics to the Linear Regression results (Section 7).")
                print("Higher R², lower MAE/MSE/RMSE indicate better performance.")

                # --- Step 8.9: Feature Importance ---
                print("\n--- Step 8.9: LightGBM Feature Importance ---")
                try:
                    # Get feature names after preprocessing (including one-hot encoded ones)
                    feature_names_processed = preprocessor.get_feature_names_out()

                    importance_df_lgbm = pd.DataFrame({
                        'Feature': feature_names_processed,
                        'Importance': lgbm_model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)

                    print("Top 20 Features by Importance:")
                    print(importance_df_lgbm.head(20))

                    # Plot top N feature importances
                    n_features_to_plot = 20
                    plt.figure(figsize=(10, max(6, n_features_to_plot * 0.3)))
                    sns.barplot(x='Importance', y='Feature', data=importance_df_lgbm.head(n_features_to_plot), palette='rocket')
                    plt.title(f'Top {n_features_to_plot} Feature Importances (LightGBM)')
                    plt.tight_layout()
                    plt.show()

                except Exception as e_imp:
                     print(f"Could not generate feature importance plot: {e_imp}")


            except Exception as e:
                 print(f"Error during LightGBM training or evaluation: {e}")

else:
    # This else corresponds to the outer RUN_SECTION_8_LIGHTGBM check
    if not ('RUN_SECTION_8_LIGHTGBM' in locals() and RUN_SECTION_8_LIGHTGBM): # Avoid printing if it was just skipped due to import error
        print("\n--- [SKIPPING] 8. Advanced Modeling: LightGBM ---")


# =============================================================================
# 9. Final Summary (Renumbered - This section always runs)
# =============================================================================
print("\n--- 9. Final Summary ---") # Renumbered
final_rows, final_cols = df_cleaned.shape
print(f"Original shape:        ({original_rows}, {original_cols})")
print(f"Columns dropped:       {original_cols - final_cols}")
# Calculating total rows dropped precisely requires summing drops from each step
# This is an approximation based on the difference from start to end
print(f"Total rows dropped:    {original_rows - final_rows}")
print(f"Final shape:           ({final_rows}, {final_cols})")

if not df_cleaned.empty and df_cleaned.isnull().sum().sum() > 0:
     print("\nWARNING: The final DataFrame still contains missing values.")
     print("Columns with remaining NaNs:")
     print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])
elif df_cleaned.empty:
     print("\nWARNING: The final DataFrame is empty.")
else:
     print("\nFinal DataFrame appears to have no missing values.")

print("\nProcessing complete. The final data state is in the 'df_cleaned' DataFrame.")