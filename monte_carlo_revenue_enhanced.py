"""
Enhanced Monte Carlo Revenue Distribution Model for Merchant Renewable Assets

Features:
- Monthly forecast expansion to hourly
- Time-varying volatility by month
- Comprehensive sensitivity analysis
- CVaR tail risk metrics
- Multi-asset market comparison
- Price component decomposition
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
import logging
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup logging
log_filename = f'log_mc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AssetData:
    """Container for historical and forecast data for a single asset."""
    name: str
    historical: pd.DataFrame
    monthly_forecasts: pd.DataFrame = None  # Monthly forecasts for 2026-2030


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 10000
    forecast_years: int = 5
    start_date: str = '2026-01-01'
    discount_rate: float = 0.07

    # Sensitivity analysis scenarios
    price_scenarios: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    volume_scenarios: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    basis_vol_scenarios: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    p_levels: List[int] = field(default_factory=lambda: [10, 25, 50, 75, 90])


class MonteCarloRevenueModel:
    """
    Enhanced Monte Carlo simulation for merchant renewable revenue.

    Key features:
    - Time-varying volatility (monthly patterns)
    - Basis risk modeling with separate uncertainty
    - Correlation preservation (gen-price, hub-busbar, RT-DA)
    - CVaR tail risk analysis
    - Comprehensive sensitivity testing
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.random_state = np.random.RandomState(42)
        self.results = {}

    def load_asset_data(self, filepath: str, sheet_name: str) -> AssetData:
        """Load historical data and monthly forecasts from Excel."""
        logger.info(f"Loading data for {sheet_name}...")
        t0 = time.time()
        df = pd.read_excel(filepath, sheet_name=sheet_name, header=8)
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        logger.info(f"  Excel loaded in {time.time()-t0:.2f}s")

        # Clean column names but keep mapping to find Gen column
        col_mapping = {}
        cleaned_cols = []
        for i, col in enumerate(df.columns):
            if not pd.isna(col):
                cleaned = str(col).strip()
                cleaned_cols.append(cleaned)
                col_mapping[cleaned] = str(col)
            else:
                cleaned_cols.append(f'Unnamed_{i}')

        # Find Gen column (could be 'Gen' or 'Gen ')
        gen_col = None
        for col in df.columns:
            if str(col).strip() == 'Gen':
                gen_col = col
                break

        # Convert to numeric
        numeric_cols = ['RT Busbar', 'RT Hub', 'DA Busbar', 'DA Hub', 'HE']
        if gen_col:
            numeric_cols.append(gen_col)

        for col in df.columns:
            col_clean = str(col).strip()
            if col_clean in numeric_cols or col == gen_col:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Historical data (has generation values)
        if gen_col:
            historical = df[df[gen_col].notna()].copy()
            historical = historical.rename(columns={gen_col: 'Gen'})
        else:
            historical = df.copy()

        # Monthly forecasts (2026-2030) - column 10 has forecast dates
        monthly_forecasts = None
        logger.info(f"  Checking for monthly forecasts...")
        logger.info(f"    'Peak' in columns: {'Peak' in df.columns}")
        if 'Peak' in df.columns:
            logger.info(f"    Peak has data: {df['Peak'].notna().any()}")
            logger.info(f"    Rows with Peak data: {df['Peak'].notna().sum()}")

        if 'Peak' in df.columns and df['Peak'].notna().any():
            fwd_data = df[df['Peak'].notna()].copy()
            logger.info(f"    Forward data rows: {len(fwd_data)}")

            # Column 10 contains forecast month dates (column header might be NaT or Unnamed_10)
            # Find the date column - it's around index 10
            date_col = None
            for col in fwd_data.columns:
                if pd.isna(col) and isinstance(col, type(pd.NaT)):
                    # Try to see if this column has dates
                    try:
                        test_dates = pd.to_datetime(fwd_data[col], errors='coerce')
                        if test_dates.notna().any() and test_dates.dt.year.min() >= 2026:
                            date_col = col
                            break
                    except:
                        continue

            # Fallback: check for Unnamed_10
            if date_col is None and 'Unnamed_10' in fwd_data.columns:
                date_col = 'Unnamed_10'

            logger.info(f"    Found date column: {date_col}")
            if date_col is not None:
                fwd_data['ForecastDate'] = pd.to_datetime(fwd_data[date_col], errors='coerce')
                fwd_data = fwd_data[fwd_data['ForecastDate'].notna()].copy()
                logger.info(f"    Rows with valid forecast dates: {len(fwd_data)}")

                # Extract forecast columns
                forecast_cols = {
                    'ForecastDate': fwd_data['ForecastDate'],
                    'Peak_Price': pd.to_numeric(fwd_data['Peak'], errors='coerce'),
                    'OffPeak_Price': pd.to_numeric(fwd_data['Off Peak'], errors='coerce'),
                }

                # Generation forecasts if available
                if 'Exp.Gen.Peak' in fwd_data.columns:
                    forecast_cols['Gen_Peak'] = pd.to_numeric(fwd_data['Exp.Gen.Peak'], errors='coerce')
                    forecast_cols['Gen_OffPeak'] = pd.to_numeric(fwd_data['Exp.Gen.Off Peak'], errors='coerce')

                # Price forecasts by market
                if 'RT Hub Forecast' in fwd_data.columns:
                    forecast_cols['RT_Hub'] = pd.to_numeric(fwd_data['RT Hub Forecast'], errors='coerce')
                    forecast_cols['RT_Busbar'] = pd.to_numeric(fwd_data['RT Busbar Forecast'], errors='coerce')
                    forecast_cols['DA_Hub'] = pd.to_numeric(fwd_data['DA Hub Forecast'], errors='coerce')
                    forecast_cols['DA_Busbar'] = pd.to_numeric(fwd_data['DA Busbar Forecast'], errors='coerce')

                monthly_forecasts = pd.DataFrame(forecast_cols)
                monthly_forecasts['month'] = monthly_forecasts['ForecastDate'].dt.month
                monthly_forecasts['year'] = monthly_forecasts['ForecastDate'].dt.year

                logger.info(f"  Monthly forecast columns loaded: {list(monthly_forecasts.columns)}")
                logger.info(f"  Forecast date range: {monthly_forecasts['ForecastDate'].min()} to {monthly_forecasts['ForecastDate'].max()}")

        return AssetData(
            name=sheet_name,
            historical=historical,
            monthly_forecasts=monthly_forecasts
        )

    def calculate_historical_statistics(self, asset: AssetData) -> Dict:
        """Calculate comprehensive statistics from historical data."""
        logger.info("Calculating historical statistics...")
        t0 = time.time()
        hist = asset.historical.copy()
        hist['hour'] = hist['HE']
        hist['month'] = hist['Date'].dt.month
        hist['year'] = hist['Date'].dt.year
        hist['is_peak'] = hist['P/OP'].str.strip() == 'P' if 'P/OP' in hist.columns else False

        stats_dict = {
            'generation': {},
            'prices': {},
            'correlations': {},
            'volatility': {}
        }

        # Generation statistics
        stats_dict['generation']['mean'] = hist['Gen'].mean()
        stats_dict['generation']['std'] = hist['Gen'].std()
        stats_dict['generation']['by_month'] = hist.groupby('month')['Gen'].agg(['mean', 'std']).to_dict()
        stats_dict['generation']['by_hour'] = hist.groupby('hour')['Gen'].agg(['mean', 'std']).to_dict()
        stats_dict['generation']['by_peak'] = hist.groupby('is_peak')['Gen'].agg(['mean', 'std']).to_dict()

        # Price statistics with time-varying volatility
        for market in ['RT Busbar', 'RT Hub', 'DA Busbar', 'DA Hub']:
            if market in hist.columns:
                market_key = market.replace(' ', '_')
                stats_dict['prices'][market_key] = {
                    'mean': hist[market].mean(),
                    'std': hist[market].std(),
                    'by_month': hist.groupby('month')[market].agg(['mean', 'std']).to_dict(),
                    'by_hour': hist.groupby('hour')[market].agg(['mean', 'std']).to_dict(),
                    'negative_price_freq': (hist[market] < 0).mean(),
                    'negative_price_hours': (hist[market] < 0).sum(),
                }

                # Monthly volatility (time-varying)
                monthly_vol = hist.groupby('month')[market].std()
                stats_dict['volatility'][f'{market_key}_monthly'] = monthly_vol.to_dict()

        # Basis statistics
        if 'RT Busbar' in hist.columns and 'RT Hub' in hist.columns:
            hist['RT_Basis'] = hist['RT Busbar'] - hist['RT Hub']
            stats_dict['basis'] = {
                'RT_mean': hist['RT_Basis'].mean(),
                'RT_std': hist['RT_Basis'].std(),
                'RT_by_month': hist.groupby('month')['RT_Basis'].agg(['mean', 'std']).to_dict()
            }

        if 'DA Busbar' in hist.columns and 'DA Hub' in hist.columns:
            hist['DA_Basis'] = hist['DA Busbar'] - hist['DA Hub']
            if 'basis' not in stats_dict:
                stats_dict['basis'] = {}
            stats_dict['basis']['DA_mean'] = hist['DA_Basis'].mean()
            stats_dict['basis']['DA_std'] = hist['DA_Basis'].std()
            stats_dict['basis']['DA_by_month'] = hist.groupby('month')['DA_Basis'].agg(['mean', 'std']).to_dict()

        # Correlations
        for market in ['RT Busbar', 'RT Hub', 'DA Busbar', 'DA Hub']:
            if market in hist.columns:
                stats_dict['correlations'][f'Gen_{market}'] = hist['Gen'].corr(hist[market])

        if 'RT Busbar' in hist.columns and 'RT Hub' in hist.columns:
            stats_dict['correlations']['RT_Basis_Hub'] = hist['RT_Basis'].corr(hist['RT Hub'])
        if 'DA Busbar' in hist.columns and 'DA Hub' in hist.columns:
            stats_dict['correlations']['DA_Basis_Hub'] = hist['DA_Basis'].corr(hist['DA Hub'])
        if 'DA Hub' in hist.columns and 'RT Hub' in hist.columns:
            stats_dict['correlations']['DA_RT_Hub'] = hist['DA Hub'].corr(hist['RT Hub'])

        logger.info(f"  Statistics calculated in {time.time()-t0:.2f}s")
        return stats_dict

    def expand_monthly_to_hourly(self, asset: AssetData, stats: Dict,
                                  column: str, apply_peak_offpeak: bool = False) -> np.ndarray:
        """
        Expand monthly forecast to hourly using historical patterns (VECTORIZED).

        Args:
            column: Column name in monthly_forecasts ('Gen_Peak', 'RT_Hub', etc.)
            apply_peak_offpeak: If True, use Peak/OffPeak columns separately
        """
        logger.info(f"  Expanding {column} to hourly...")
        t0 = time.time()

        hours_per_year = 8760
        total_hours = hours_per_year * self.config.forecast_years
        hourly_forecast = np.zeros(total_hours)

        if asset.monthly_forecasts is None:
            logger.info(f"    monthly_forecasts is None, using zeros")
            return hourly_forecast

        if column not in asset.monthly_forecasts.columns:
            logger.info(f"    Column '{column}' not in monthly_forecasts. Available columns: {list(asset.monthly_forecasts.columns)}")
            return hourly_forecast

        hist = asset.historical.copy()
        hist['hour'] = hist['HE']
        hist['month'] = hist['Date'].dt.month
        hist['is_peak'] = hist['P/OP'].str.strip() == 'P' if 'P/OP' in hist.columns else False

        # Determine historical column to use for patterns
        if column.startswith('Gen'):
            hist_col = 'Gen'
        elif 'RT_Hub' in column:
            hist_col = 'RT Hub'
        elif 'RT_Busbar' in column:
            hist_col = 'RT Busbar'
        elif 'DA_Hub' in column:
            hist_col = 'DA Hub'
        elif 'DA_Busbar' in column:
            hist_col = 'DA Busbar'
        else:
            return hourly_forecast

        if hist_col not in hist.columns:
            return hourly_forecast

        # VECTORIZED: Pre-calculate shape factors lookup table
        # Shape factor = typical hourly pattern within each month
        monthly_avg = hist.groupby('month')[hist_col].mean()
        hist['monthly_avg'] = hist['month'].map(monthly_avg)
        hist['shape_factor'] = hist[hist_col] / hist['monthly_avg'].replace(0, 1)

        # Create lookup table: (month, hour) -> median shape factor
        if apply_peak_offpeak:
            shape_lookup = hist.groupby(['month', 'hour', 'is_peak'])['shape_factor'].median().to_dict()
        else:
            shape_lookup = hist.groupby(['month', 'hour'])['shape_factor'].median().to_dict()

        # VECTORIZED: Create time index arrays
        start_date = pd.Timestamp(self.config.start_date)
        time_index = pd.date_range(start=start_date, periods=total_hours, freq='H')
        months = time_index.month.values
        hours = time_index.hour.values
        is_peak_hours = (hours >= 7) & (hours < 23)

        # Get monthly forecast values - average across years for each month
        # (We have 2026-2030, so multiple values per month)
        monthly_fc = asset.monthly_forecasts.groupby('month').mean(numeric_only=True)

        # VECTORIZED: Build hourly forecast array
        for month in range(1, 13):
            month_mask = months == month

            if month not in monthly_fc.index:
                continue

            # Get forecast value for this month
            if apply_peak_offpeak and column.startswith('Gen'):
                # Peak vs Off-Peak generation
                peak_mask = month_mask & is_peak_hours
                offpeak_mask = month_mask & ~is_peak_hours

                if 'Gen_Peak' in monthly_fc.columns:
                    monthly_value_peak = float(monthly_fc.loc[month, 'Gen_Peak'])
                    monthly_value_offpeak = float(monthly_fc.loc[month, 'Gen_OffPeak']) if 'Gen_OffPeak' in monthly_fc.columns else monthly_value_peak
                else:
                    monthly_value_peak = float(monthly_fc.loc[month, column])
                    monthly_value_offpeak = monthly_value_peak

                # Apply shape factors
                for hour in range(24):
                    hour_peak_mask = peak_mask & (hours == hour)
                    hour_offpeak_mask = offpeak_mask & (hours == hour)

                    if hour_peak_mask.any():
                        shape = float(shape_lookup.get((month, hour + 1, True), 1.0))
                        hourly_forecast[hour_peak_mask] = monthly_value_peak * shape

                    if hour_offpeak_mask.any():
                        shape = float(shape_lookup.get((month, hour + 1, False), 1.0))
                        hourly_forecast[hour_offpeak_mask] = monthly_value_offpeak * shape
            else:
                # Regular price forecasts
                monthly_value = float(monthly_fc.loc[month, column])

                for hour in range(24):
                    hour_mask = month_mask & (hours == hour)
                    if hour_mask.any():
                        shape = float(shape_lookup.get((month, hour + 1), 1.0))
                        hourly_forecast[hour_mask] = monthly_value * shape

        logger.info(f"    Expansion completed in {time.time()-t0:.2f}s")
        return hourly_forecast

    def simulate_generation(self, asset: AssetData, stats: Dict,
                           volume_multiplier: float = 1.0,
                           base_gen_hourly: np.ndarray = None) -> np.ndarray:
        """
        Simulate generation with time-varying uncertainty.

        Args:
            base_gen_hourly: Pre-expanded hourly generation forecast (optional, for reuse)

        Returns: array of shape (n_simulations, n_hours)
        """
        logger.info("Simulating generation...")
        t0 = time.time()

        hours_per_year = 8760
        total_hours = hours_per_year * self.config.forecast_years

        # Expand monthly forecasts to hourly (or reuse if provided)
        if base_gen_hourly is None:
            base_gen = self.expand_monthly_to_hourly(asset, stats, 'Gen_Peak', apply_peak_offpeak=True)
        else:
            base_gen = base_gen_hourly.copy()
            logger.info("  Reusing pre-expanded hourly generation forecast")

        base_gen = base_gen * volume_multiplier

        # If no forecast, use historical mean
        if base_gen.sum() == 0:
            base_gen = np.full(total_hours, stats['generation']['mean'])

        # Time-varying volatility by month
        start_date = pd.Timestamp(self.config.start_date)
        monthly_std = np.array([stats['generation']['by_month']['std'].get(
            (start_date + pd.Timedelta(hours=i)).month, stats['generation']['std'])
            for i in range(total_hours)])

        # Generate simulations
        simulated_gen = np.zeros((self.config.n_simulations, total_hours))
        for i in range(self.config.n_simulations):
            noise = self.random_state.normal(0, 1, total_hours)
            simulated_gen[i] = np.maximum(0, base_gen + noise * monthly_std * 0.3)

        logger.info(f"  Generation simulated in {time.time()-t0:.2f}s")
        return simulated_gen

    def simulate_prices(self, asset: AssetData, stats: Dict, market: str,
                       price_multiplier: float = 1.0,
                       basis_vol_multiplier: float = 1.0,
                       neg_price_multiplier: float = 1.0,
                       base_price_hourly: np.ndarray = None) -> np.ndarray:
        """
        Simulate prices with time-varying volatility and basis risk.

        Args:
            market: 'RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar'
            base_price_hourly: Pre-expanded hourly price forecast (optional, for reuse)

        Returns: array of shape (n_simulations, n_hours)
        """
        hours_per_year = 8760
        total_hours = hours_per_year * self.config.forecast_years

        # Expand monthly forecast to hourly (or reuse if provided)
        if base_price_hourly is None:
            base_price = self.expand_monthly_to_hourly(asset, stats, market)
        else:
            base_price = base_price_hourly.copy()

        base_price = base_price * price_multiplier

        # Fallback to historical mean if no forecast
        if base_price.sum() == 0:
            base_price = np.full(total_hours, stats['prices'][market]['mean'])

        # Time-varying volatility
        start_date = pd.Timestamp(self.config.start_date)
        monthly_vol_dict = stats['volatility'].get(f'{market}_monthly', {})
        monthly_std = np.array([monthly_vol_dict.get(
            (start_date + pd.Timedelta(hours=i)).month, stats['prices'][market]['std'])
            for i in range(total_hours)])

        # Simulate prices
        simulated_prices = np.zeros((self.config.n_simulations, total_hours))
        for i in range(self.config.n_simulations):
            noise = self.random_state.normal(0, 1, total_hours)
            simulated_prices[i] = base_price + noise * monthly_std

            # Apply negative price multiplier (increase frequency)
            if neg_price_multiplier != 1.0 and neg_price_multiplier > 0:
                neg_mask = simulated_prices[i] < 0
                simulated_prices[i][neg_mask] = simulated_prices[i][neg_mask] * neg_price_multiplier

        return simulated_prices

    def simulate_correlated_prices(self, asset: AssetData, stats: Dict,
                                   generation: np.ndarray,
                                   price_multiplier: float = 1.0,
                                   basis_vol_multiplier: float = 1.0,
                                   base_prices_hourly: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Simulate all prices with preserved correlations and basis risk.

        Args:
            base_prices_hourly: Dict of pre-expanded hourly forecasts {'RT_Hub': array, 'DA_Hub': array}

        Returns dict with keys: 'RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar'
        """
        n_sim, n_hours = generation.shape

        # Get correlation structure
        corr_gen_rt = stats['correlations'].get('Gen_RT Hub', -0.2)
        corr_gen_da = stats['correlations'].get('Gen_DA Hub', -0.2)
        corr_rt_da = stats['correlations'].get('DA_RT_Hub', 0.9)

        # Simulate hub prices first (these are the reference)
        rt_hub_base = base_prices_hourly.get('RT_Hub') if base_prices_hourly else None
        da_hub_base = base_prices_hourly.get('DA_Hub') if base_prices_hourly else None

        rt_hub = self.simulate_prices(asset, stats, 'RT_Hub', price_multiplier, base_price_hourly=rt_hub_base)
        da_hub = self.simulate_prices(asset, stats, 'DA_Hub', price_multiplier, base_price_hourly=da_hub_base)

        # Simulate basis spreads separately with their own uncertainty
        rt_basis_mean = stats['basis'].get('RT_mean', 0)
        rt_basis_std = stats['basis'].get('RT_std', 10) * basis_vol_multiplier
        da_basis_mean = stats['basis'].get('DA_mean', 0)
        da_basis_std = stats['basis'].get('DA_std', 10) * basis_vol_multiplier

        # Vectorized basis generation (much faster than looping)
        rt_basis = rt_basis_mean + self.random_state.normal(0, rt_basis_std, (n_sim, n_hours))
        da_basis = da_basis_mean + self.random_state.normal(0, da_basis_std, (n_sim, n_hours))

        # Busbar = Hub + Basis
        rt_busbar = rt_hub + rt_basis
        da_busbar = da_hub + da_basis

        return {
            'RT_Hub': rt_hub,
            'RT_Busbar': rt_busbar,
            'DA_Hub': da_hub,
            'DA_Busbar': da_busbar,
            'RT_Basis': rt_basis,
            'DA_Basis': da_basis
        }

    def calculate_revenue(self, generation: np.ndarray, prices: np.ndarray,
                         negative_price_protection: bool = False) -> np.ndarray:
        """Calculate revenue for each simulation."""
        revenue = generation * prices

        if negative_price_protection:
            revenue = np.maximum(revenue, 0)

        return revenue.sum(axis=1)  # Sum over hours for each simulation

    def calculate_risk_metrics(self, revenue_distribution: np.ndarray) -> Dict:
        """Calculate comprehensive risk metrics including CVaR."""
        metrics = {}

        # Percentiles
        for p in self.config.p_levels:
            metrics[f'P{p}'] = np.percentile(revenue_distribution, p)

        # Standard metrics
        metrics['mean'] = revenue_distribution.mean()
        metrics['std'] = revenue_distribution.std()
        metrics['cv'] = metrics['std'] / metrics['mean'] if metrics['mean'] != 0 else 0

        # Tail risk: CVaR (Conditional Value at Risk)
        # CVaR at 10% = average of worst 10% outcomes
        cvar_10_threshold = np.percentile(revenue_distribution, 10)
        worst_10pct = revenue_distribution[revenue_distribution <= cvar_10_threshold]
        metrics['CVaR_10'] = worst_10pct.mean()

        # Downside deviation (semi-variance below median)
        median = metrics['P50']
        downside = revenue_distribution[revenue_distribution < median]
        metrics['downside_deviation'] = np.sqrt(((downside - median) ** 2).mean()) if len(downside) > 0 else 0

        return metrics

    def run_simulation(self, asset: AssetData, stats: Dict,
                      volume_mult: float = 1.0,
                      price_mult: float = 1.0,
                      basis_vol_mult: float = 1.0,
                      neg_price_mult: float = 1.0,
                      base_gen_hourly: np.ndarray = None,
                      base_prices_hourly: Dict[str, np.ndarray] = None) -> Dict:
        """Run full Monte Carlo simulation for one asset."""
        logger.info(f"Running Monte Carlo ({self.config.n_simulations} sims)...")
        t0_total = time.time()

        # Simulate generation
        generation = self.simulate_generation(asset, stats, volume_mult, base_gen_hourly=base_gen_hourly)

        # Simulate all prices with correlations and basis risk
        logger.info("Simulating prices with correlations...")
        t0 = time.time()
        prices = self.simulate_correlated_prices(asset, stats, generation,
                                                 price_mult, basis_vol_mult,
                                                 base_prices_hourly=base_prices_hourly)
        logger.info(f"  Prices simulated in {time.time()-t0:.2f}s")

        # Calculate revenues for each market scenario
        logger.info("Calculating revenues for each market...")
        results = {}
        for market in ['RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar']:
            logger.info(f"  Processing {market}...")
            t_market = time.time()
            market_prices = prices[market]

            # Merchant revenue (with and without negative price protection)
            rev_merchant = self.calculate_revenue(generation, market_prices, negative_price_protection=False)
            rev_protected = self.calculate_revenue(generation, market_prices, negative_price_protection=True)
            logger.info(f"    Revenue calculated in {time.time()-t_market:.2f}s")

            # Risk metrics
            t_risk = time.time()
            metrics_merchant = self.calculate_risk_metrics(rev_merchant)
            metrics_protected = self.calculate_risk_metrics(rev_protected)
            logger.info(f"    Risk metrics calculated in {time.time()-t_risk:.2f}s")

            # Fixed prices at different P-levels
            total_gen = generation.sum(axis=1).mean()  # Expected total generation
            fixed_prices = {f'P{p}': metrics_merchant[f'P{p}'] / total_gen
                          for p in self.config.p_levels}

            # Price component breakdown (for P25)
            hub_market = market.replace('Busbar', 'Hub')
            mean_hub_price = prices[hub_market].mean()

            if 'Busbar' in market:
                basis_key = 'RT_Basis' if 'RT' in market else 'DA_Basis'
                mean_basis = prices[basis_key].mean()
            else:
                mean_basis = 0.0

            risk_premium = fixed_prices['P25'] - fixed_prices['P50']

            results[market] = {
                'revenue_distribution': rev_merchant,
                'revenue_protected': rev_protected,
                'metrics_merchant': metrics_merchant,
                'metrics_protected': metrics_protected,
                'fixed_prices': fixed_prices,
                'total_generation_mwh': total_gen,
                'price_components': {
                    'hub_price': mean_hub_price,
                    'basis': mean_basis,
                    'risk_premium': risk_premium,
                    'total_p25': fixed_prices['P25']
                },
                'negative_price_impact': (rev_protected.mean() - rev_merchant.mean()) / rev_merchant.mean() * 100
            }

        logger.info(f"  Total simulation time: {time.time()-t0_total:.2f}s")
        return results

    def sensitivity_analysis(self, asset: AssetData, stats: Dict) -> pd.DataFrame:
        """Run comprehensive sensitivity analysis."""
        logger.info("Starting sensitivity analysis...")
        t0_sens = time.time()

        # PRE-EXPAND hourly forecasts once (this is the slow part)
        logger.info("  Pre-expanding monthly forecasts to hourly (one-time)...")
        t_expand = time.time()
        base_gen_hourly = self.expand_monthly_to_hourly(asset, stats, 'Gen_Peak', apply_peak_offpeak=True)
        base_prices_hourly = {
            'RT_Hub': self.expand_monthly_to_hourly(asset, stats, 'RT_Hub'),
            'DA_Hub': self.expand_monthly_to_hourly(asset, stats, 'DA_Hub'),
        }
        logger.info(f"  Expansion completed in {time.time()-t_expand:.2f}s - will reuse for all scenarios")

        sensitivity_results = []
        logger.info("  Running base case...")
        base_results = self.run_simulation(asset, stats,
                                          base_gen_hourly=base_gen_hourly,
                                          base_prices_hourly=base_prices_hourly)
        base_p25 = base_results['DA_Hub']['fixed_prices']['P25']

        # Price sensitivity
        logger.info(f"  Testing price scenarios: {self.config.price_scenarios}")
        for price_mult in self.config.price_scenarios:
            results = self.run_simulation(asset, stats, price_mult=price_mult,
                                         base_gen_hourly=base_gen_hourly,
                                         base_prices_hourly=base_prices_hourly)
            p25 = results['DA_Hub']['fixed_prices']['P25']
            sensitivity_results.append({
                'Parameter': 'Forward Prices',
                'Scenario': f'{price_mult*100:.0f}%',
                'Multiplier': price_mult,
                'P25_Fixed_Price': p25,
                'Change_from_Base': p25 - base_p25,
                'Pct_Change': (p25 - base_p25) / base_p25 * 100
            })

        # Volume sensitivity
        for vol_mult in self.config.volume_scenarios:
            results = self.run_simulation(asset, stats, volume_mult=vol_mult,
                                         base_gen_hourly=base_gen_hourly,
                                         base_prices_hourly=base_prices_hourly)
            p25 = results['DA_Hub']['fixed_prices']['P25']
            sensitivity_results.append({
                'Parameter': 'Generation Volume',
                'Scenario': f'{vol_mult*100:.0f}%',
                'Multiplier': vol_mult,
                'P25_Fixed_Price': p25,
                'Change_from_Base': p25 - base_p25,
                'Pct_Change': (p25 - base_p25) / base_p25 * 100
            })

        # Basis volatility sensitivity
        for basis_mult in self.config.basis_vol_scenarios:
            results = self.run_simulation(asset, stats, basis_vol_mult=basis_mult,
                                         base_gen_hourly=base_gen_hourly,
                                         base_prices_hourly=base_prices_hourly)
            p25 = results['DA_Busbar']['fixed_prices']['P25']
            sensitivity_results.append({
                'Parameter': 'Basis Volatility',
                'Scenario': f'{basis_mult*100:.0f}%',
                'Multiplier': basis_mult,
                'P25_Fixed_Price': p25,
                'Change_from_Base': p25 - base_p25,
                'Pct_Change': (p25 - base_p25) / base_p25 * 100
            })

        # Negative price sensitivity
        for neg_mult in [1.0, 2.0, 3.0]:
            results = self.run_simulation(asset, stats, neg_price_mult=neg_mult,
                                         base_gen_hourly=base_gen_hourly,
                                         base_prices_hourly=base_prices_hourly)
            p25 = results['RT_Hub']['fixed_prices']['P25']
            sensitivity_results.append({
                'Parameter': 'Negative Price Frequency',
                'Scenario': f'{neg_mult:.0f}x Historical',
                'Multiplier': neg_mult,
                'P25_Fixed_Price': p25,
                'Change_from_Base': p25 - base_p25,
                'Pct_Change': (p25 - base_p25) / base_p25 * 100
            })

        logger.info(f"  Sensitivity analysis completed in {time.time()-t0_sens:.2f}s")
        return pd.DataFrame(sensitivity_results)

    def analyze_asset(self, filepath: str, sheet_name: str) -> Dict:
        """Complete analysis for one asset."""
        print(f"\n{'='*70}")
        print(f"Analyzing {sheet_name}")
        print(f"{'='*70}")

        # Load data
        asset = self.load_asset_data(filepath, sheet_name)
        print(f"Loaded historical data: {len(asset.historical)} hours")
        if asset.monthly_forecasts is not None:
            print(f"Loaded monthly forecasts: {len(asset.monthly_forecasts)} months")

        # Calculate statistics
        stats = self.calculate_historical_statistics(asset)
        print(f"Historical mean generation: {stats['generation']['mean']:.2f} MW")
        print(f"Historical mean DA Hub price: ${stats['prices']['DA_Hub']['mean']:.2f}/MWh")

        # Run base simulation
        print(f"\nRunning {self.config.n_simulations} Monte Carlo simulations...")
        results = self.run_simulation(asset, stats)

        # Sensitivity analysis
        print("Running sensitivity analysis...")
        sensitivity = self.sensitivity_analysis(asset, stats)

        # Market comparison metrics
        market_metrics = {
            'revenue_cv': results['DA_Hub']['metrics_merchant']['cv'],
            'basis_risk_rt': stats['basis'].get('RT_std', 0),
            'basis_risk_da': stats['basis'].get('DA_std', 0),
            'negative_price_freq': stats['prices']['RT_Hub']['negative_price_freq'],
            'cvar_10': results['DA_Hub']['metrics_merchant']['CVaR_10'],
        }

        return {
            'asset': asset,
            'stats': stats,
            'results': results,
            'sensitivity': sensitivity,
            'market_metrics': market_metrics
        }

    def save_results(self, analysis: Dict, output_dir: Path):
        """Save all results to CSV and visualizations."""
        asset_name = analysis['asset'].name
        results = analysis['results']

        # Summary table
        summary_data = []
        for market in ['RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar']:
            res = results[market]
            components = res['price_components']

            row = {
                'Asset': asset_name,
                'Market': market,
                'Total_Generation_MWh': res['total_generation_mwh'],
            }

            # Add P-level prices
            for p in self.config.p_levels:
                row[f'P{p}_Fixed_Price'] = res['fixed_prices'][f'P{p}']

            # Price components
            row['Hub_Price_Component'] = components['hub_price']
            row['Basis_Component'] = components['basis']
            row['Risk_Premium'] = components['risk_premium']

            # Risk metrics
            row['Revenue_CV'] = res['metrics_merchant']['cv']
            row['CVaR_10'] = res['metrics_merchant']['CVaR_10']
            row['Downside_Deviation'] = res['metrics_merchant']['downside_deviation']
            row['Neg_Price_Protection_Value_Pct'] = res['negative_price_impact']

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / f'{asset_name}_summary.csv', index=False)
        print(f"Saved: {output_dir / f'{asset_name}_summary.csv'}")

        # Sensitivity analysis
        analysis['sensitivity'].to_csv(output_dir / f'{asset_name}_sensitivity.csv', index=False)
        print(f"Saved: {output_dir / f'{asset_name}_sensitivity.csv'}")

        # Visualizations
        self.create_visualizations(analysis, output_dir)

    def create_visualizations(self, analysis: Dict, output_dir: Path):
        """Create comprehensive visualizations."""
        asset_name = analysis['asset'].name
        results = analysis['results']

        # 1. Revenue distributions with P-levels
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{asset_name}: Revenue Distributions by Market', fontsize=16, fontweight='bold')

        markets = ['RT_Hub', 'RT_Busbar', 'DA_Hub', 'DA_Busbar']
        for idx, market in enumerate(markets):
            ax = axes[idx // 2, idx % 2]
            dist = results[market]['revenue_distribution'] / 1e6  # Convert to $M

            ax.hist(dist, bins=100, alpha=0.7, edgecolor='black', density=True)

            # Add P-level lines
            colors = {'P10': 'red', 'P25': 'orange', 'P50': 'green', 'P75': 'blue', 'P90': 'purple'}
            for p in [10, 25, 50, 75, 90]:
                val = results[market]['metrics_merchant'][f'P{p}'] / 1e6
                ax.axvline(val, color=colors[f'P{p}'], linestyle='--', linewidth=2,
                          label=f'P{p}: ${val:.1f}M')

            # CVaR
            cvar = results[market]['metrics_merchant']['CVaR_10'] / 1e6
            ax.axvline(cvar, color='darkred', linestyle=':', linewidth=2,
                      label=f'CVaR 10%: ${cvar:.1f}M')

            ax.set_xlabel('5-Year Revenue ($M)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{market.replace("_", " ")}\nCV: {results[market]["metrics_merchant"]["cv"]:.2%}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{asset_name}_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'{asset_name}_distributions.png'}")
        plt.close()

        # 2. Sensitivity tornado chart
        sens_df = analysis['sensitivity']
        base_case = sens_df[sens_df['Multiplier'] == 1.0].iloc[0] if len(sens_df[sens_df['Multiplier'] == 1.0]) > 0 else None

        if base_case is not None:
            fig, ax = plt.subplots(figsize=(12, 8))

            params = sens_df['Parameter'].unique()
            y_pos = np.arange(len(params))

            for i, param in enumerate(params):
                param_data = sens_df[sens_df['Parameter'] == param].sort_values('Multiplier')
                if len(param_data) >= 2:
                    low = param_data.iloc[0]['P25_Fixed_Price']
                    high = param_data.iloc[-1]['P25_Fixed_Price']
                    base = base_case['P25_Fixed_Price']

                    ax.barh(i, low - base, left=base, color='red', alpha=0.6)
                    ax.barh(i, high - base, left=base, color='green', alpha=0.6)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.set_xlabel('P25 Fixed Price ($/MWh)')
            ax.set_title(f'{asset_name}: Sensitivity Analysis - Tornado Chart', fontweight='bold')
            ax.axvline(base_case['P25_Fixed_Price'], color='black', linestyle='--', linewidth=2, label='Base Case')
            ax.legend()
            ax.grid(alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(output_dir / f'{asset_name}_tornado.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir / f'{asset_name}_tornado.png'}")
            plt.close()


def main():
    """Run analysis for all assets."""

    # Configuration
    config = SimulationConfig(
        n_simulations=10000,
        forecast_years=5,
        start_date='2026-01-01'
    )

    model = MonteCarloRevenueModel(config)

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Analyze all assets
    filepath = 'HackathonDataset_WithGenAndPriceForecasts.xlsx'
    all_results = {}

    for sheet_name in ['ERCOT', 'MISO', 'CAISO']:
        try:
            analysis = model.analyze_asset(filepath, sheet_name)
            all_results[sheet_name] = analysis
            model.save_results(analysis, results_dir)
        except Exception as e:
            print(f"Error analyzing {sheet_name}: {e}")
            import traceback
            traceback.print_exc()

    # Market comparison summary
    print("\n" + "="*70)
    print("MARKET COMPARISON SUMMARY")
    print("="*70)

    comparison_data = []
    for asset_name, analysis in all_results.items():
        metrics = analysis['market_metrics']
        comparison_data.append({
            'Asset': asset_name,
            'Revenue_CV': metrics['revenue_cv'],
            'RT_Basis_Risk_StdDev': metrics['basis_risk_rt'],
            'DA_Basis_Risk_StdDev': metrics['basis_risk_da'],
            'Negative_Price_Freq_Pct': metrics['negative_price_freq'] * 100,
            'CVaR_10_Million': metrics['cvar_10'] / 1e6,
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(results_dir / 'market_comparison.csv', index=False)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {results_dir / 'market_comparison.csv'}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir.absolute()}")
    print("\nKey outputs:")
    print("- *_summary.csv: Detailed results by market scenario")
    print("- *_sensitivity.csv: Sensitivity analysis")
    print("- *_distributions.png: Revenue distribution charts")
    print("- *_tornado.png: Tornado sensitivity charts")
    print("- market_comparison.csv: Cross-asset comparison")


if __name__ == '__main__':
    main()
