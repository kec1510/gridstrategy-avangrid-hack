"""
Analyze historical forecast errors to calibrate Monte Carlo uncertainty parameters.

This script extracts variance, correlation, and distributional properties from
2022-2024 historical data with generation and price forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Load ERCOT data
print("Loading ERCOT historical data with forecasts...")
df = pd.read_excel('HackathonDataset_WithGenAndPriceForecasts.xlsx',
                   sheet_name='ERCOT', header=8)

# Fix column names (first row contains actual column names)
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# Convert to appropriate types
df['Date'] = pd.to_datetime(df['Date'])
df['HE'] = pd.to_numeric(df['HE'], errors='coerce')

# Numeric columns
numeric_cols = ['Gen ', 'RT Busbar', 'RT Hub', 'DA Busbar', 'DA Hub',
                'RT Busbar Forecast', 'RT Hub Forecast',
                'DA Busbar Forecast', 'DA Hub Forecast']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter to rows with forecast data
df_forecast = df[df['RT Hub Forecast'].notna()].copy()

print(f"\nHistorical period with forecasts: {df_forecast['Date'].min()} to {df_forecast['Date'].max()}")
print(f"Total hours: {len(df_forecast)}")

# Calculate forecast errors
print("\n" + "="*70)
print("CALCULATING FORECAST ERRORS")
print("="*70)

# Generation forecast error (note: generation forecast is monthly, need to check if available)
if 'Exp.Gen.Peak' in df.columns and 'Exp.Gen.Off Peak' in df.columns:
    # Monthly generation forecasts - skip detailed error analysis for now
    print("\nGeneration forecasts are monthly - will use for forward period")

# Price forecast errors
df_forecast['RT_Hub_Error'] = df_forecast['RT Hub'] - df_forecast['RT Hub Forecast']
df_forecast['RT_Busbar_Error'] = df_forecast['RT Busbar'] - df_forecast['RT Busbar Forecast']
df_forecast['DA_Hub_Error'] = df_forecast['DA Hub'] - df_forecast['DA Hub Forecast']
df_forecast['DA_Busbar_Error'] = df_forecast['DA Busbar'] - df_forecast['DA Busbar Forecast']

# Basis errors (actual basis vs forecast basis)
df_forecast['RT_Basis_Actual'] = df_forecast['RT Busbar'] - df_forecast['RT Hub']
df_forecast['RT_Basis_Forecast'] = df_forecast['RT Busbar Forecast'] - df_forecast['RT Hub Forecast']
df_forecast['RT_Basis_Error'] = df_forecast['RT_Basis_Actual'] - df_forecast['RT_Basis_Forecast']

df_forecast['DA_Basis_Actual'] = df_forecast['DA Busbar'] - df_forecast['DA Hub']
df_forecast['DA_Basis_Forecast'] = df_forecast['DA Busbar Forecast'] - df_forecast['DA Hub Forecast']
df_forecast['DA_Basis_Error'] = df_forecast['DA_Basis_Actual'] - df_forecast['DA_Basis_Forecast']

# Summary statistics
print("\nPRICE FORECAST ERROR STATISTICS ($/MWh):")
print("-" * 70)

error_stats = {}
for market in ['RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar']:
    error_col = f'{market}_Error'
    stats = {
        'mean': df_forecast[error_col].mean(),
        'std': df_forecast[error_col].std(),
        'median': df_forecast[error_col].median(),
        'mae': df_forecast[error_col].abs().mean(),
        'rmse': np.sqrt((df_forecast[error_col]**2).mean()),
        'q25': df_forecast[error_col].quantile(0.25),
        'q75': df_forecast[error_col].quantile(0.75),
    }
    error_stats[market] = stats

    print(f"\n{market}:")
    print(f"  Mean Error (bias):     {stats['mean']:>8.2f} $/MWh")
    print(f"  Std Dev:               {stats['std']:>8.2f} $/MWh")
    print(f"  RMSE:                  {stats['rmse']:>8.2f} $/MWh")
    print(f"  MAE:                   {stats['mae']:>8.2f} $/MWh")
    print(f"  25th/75th percentile:  {stats['q25']:>8.2f} / {stats['q75']:>8.2f} $/MWh")

print("\n" + "="*70)
print("BASIS FORECAST ERROR STATISTICS ($/MWh):")
print("="*70)

basis_error_stats = {}
for market in ['RT', 'DA']:
    error_col = f'{market}_Basis_Error'
    stats = {
        'mean': df_forecast[error_col].mean(),
        'std': df_forecast[error_col].std(),
        'rmse': np.sqrt((df_forecast[error_col]**2).mean()),
        'mae': df_forecast[error_col].abs().mean(),
    }
    basis_error_stats[market] = stats

    print(f"\n{market} Basis:")
    print(f"  Mean Error (bias):     {stats['mean']:>8.2f} $/MWh")
    print(f"  Std Dev:               {stats['std']:>8.2f} $/MWh")
    print(f"  RMSE:                  {stats['rmse']:>8.2f} $/MWh")
    print(f"  MAE:                   {stats['mae']:>8.2f} $/MWh")

# Correlation analysis
print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

# Correlation between different price forecast errors
print("\nCorrelation between price forecast errors:")
error_cols = ['RT_Hub_Error', 'RT_Busbar_Error', 'DA_Hub_Error', 'DA_Busbar_Error']
corr_matrix = df_forecast[error_cols].corr()
print(corr_matrix.round(3))

# Check if we have generation data to calculate gen-price correlation
if 'Gen ' in df_forecast.columns:
    # Calculate generation "error" as deviation from mean generation for same hour/month
    df_forecast['month'] = df_forecast['Date'].dt.month
    df_forecast['hour'] = df_forecast['HE']

    monthly_hourly_avg = df_forecast.groupby(['month', 'hour'])['Gen '].transform('mean')
    df_forecast['Gen_Deviation'] = df_forecast['Gen '] - monthly_hourly_avg

    print("\nCorrelation between Generation Deviation and Price Errors:")
    gen_price_corr = {}
    for error_col in error_cols:
        corr = df_forecast[['Gen_Deviation', error_col]].corr().iloc[0, 1]
        gen_price_corr[error_col] = corr
        print(f"  {error_col}: {corr:>7.3f}")

# Temporal patterns - do errors vary by hour or month?
print("\n" + "="*70)
print("TEMPORAL PATTERNS IN FORECAST ERRORS")
print("="*70)

df_forecast['hour'] = df_forecast['HE']
df_forecast['month'] = df_forecast['Date'].dt.month

print("\nStd Dev of RT Hub Error by Hour of Day:")
hourly_std = df_forecast.groupby('hour')['RT_Hub_Error'].std()
print(f"  Min (Hour {hourly_std.idxmin()}): {hourly_std.min():.2f} $/MWh")
print(f"  Max (Hour {hourly_std.idxmax()}): {hourly_std.max():.2f} $/MWh")
print(f"  Mean across hours:      {hourly_std.mean():.2f} $/MWh")

print("\nStd Dev of RT Hub Error by Month:")
monthly_std = df_forecast.groupby('month')['RT_Hub_Error'].std()
print(f"  Min (Month {monthly_std.idxmin()}): {monthly_std.min():.2f} $/MWh")
print(f"  Max (Month {monthly_std.idxmax()}): {monthly_std.max():.2f} $/MWh")
print(f"  Mean across months:     {monthly_std.mean():.2f} $/MWh")

# Save detailed statistics to CSV
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create summary dataframe
summary_data = []
for market in ['RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar']:
    summary_data.append({
        'Market': market,
        'Mean_Error': error_stats[market]['mean'],
        'Std_Dev': error_stats[market]['std'],
        'RMSE': error_stats[market]['rmse'],
        'MAE': error_stats[market]['mae'],
        'Q25': error_stats[market]['q25'],
        'Q75': error_stats[market]['q75'],
    })

for market in ['RT', 'DA']:
    summary_data.append({
        'Market': f'{market}_Basis',
        'Mean_Error': basis_error_stats[market]['mean'],
        'Std_Dev': basis_error_stats[market]['std'],
        'RMSE': basis_error_stats[market]['rmse'],
        'MAE': basis_error_stats[market]['mae'],
        'Q25': np.nan,
        'Q75': np.nan,
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(results_dir / 'forecast_error_statistics.csv', index=False)
print(f"\nSaved: {results_dir / 'forecast_error_statistics.csv'}")

# Save correlation matrix
corr_matrix.to_csv(results_dir / 'price_error_correlations.csv')
print(f"Saved: {results_dir / 'price_error_correlations.csv'}")

# Visualizations
print("\nCreating visualizations...")

# 1. Distribution of forecast errors
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Price Forecast Error Distributions (ERCOT 2022-2024)', fontsize=14, fontweight='bold')

for idx, market in enumerate(['RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar']):
    ax = axes[idx // 2, idx % 2]
    error_col = f'{market}_Error'

    ax.hist(df_forecast[error_col], bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(error_stats[market]['mean'], color='green', linestyle='--',
               linewidth=2, label=f"Mean: {error_stats[market]['mean']:.1f}")

    ax.set_xlabel('Forecast Error ($/MWh)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{market.replace("_", " ")} Error\n(Std: {error_stats[market]["std"]:.1f} $/MWh, RMSE: {error_stats[market]["rmse"]:.1f})')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'forecast_error_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: {results_dir / 'forecast_error_distributions.png'}")

# 2. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Between Price Forecast Errors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / 'price_error_correlations.png', dpi=300, bbox_inches='tight')
print(f"Saved: {results_dir / 'price_error_correlations.png'}")

# 3. Basis error over time
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Basis Forecast Error Over Time (ERCOT)', fontsize=14, fontweight='bold')

for idx, market in enumerate(['RT', 'DA']):
    ax = axes[idx]
    error_col = f'{market}_Basis_Error'

    # Plot rolling mean and std
    rolling_mean = df_forecast.set_index('Date')[error_col].rolling('30D').mean()
    rolling_std = df_forecast.set_index('Date')[error_col].rolling('30D').std()

    ax.fill_between(rolling_mean.index,
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.3, label='±1 Std Dev (30-day rolling)')
    ax.plot(rolling_mean.index, rolling_mean, linewidth=2, label='Mean Error (30-day rolling)')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero Error')

    ax.set_ylabel('Basis Error ($/MWh)')
    ax.set_title(f'{market} Busbar-Hub Basis Forecast Error')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'basis_error_timeseries.png', dpi=300, bbox_inches='tight')
print(f"Saved: {results_dir / 'basis_error_timeseries.png'}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings for Monte Carlo Model:")
print("1. Use empirical std dev for price uncertainty in forward simulations")
print("2. Incorporate correlation structure between RT/DA and Hub/Busbar errors")
print("3. Model basis risk separately with its own uncertainty")
print("4. Consider temporal patterns (hourly/monthly variations in forecast error)")
