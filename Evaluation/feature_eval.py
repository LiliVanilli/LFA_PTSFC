import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import shap
from scipy import stats

def calculate_correlation_matrix(df, target_col):
    """
    Calculate correlation matrix for all features with respect to target variable.
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Get correlation with target
    target_corr = corr_matrix[target_col].sort_values(ascending=False)

    # Create results dataframe
    results = pd.DataFrame({
        'feature': target_corr.index,
        'correlation': target_corr.values,
        'correlation_strength': pd.cut(abs(target_corr.values),
                                       bins=[-1, 0.1, 0.3, 0.5, 1],
                                       labels=['Weak', 'Moderate', 'Strong', 'Very Strong'])
    })

    return results


def analyze_feature_importance(X, y, feature_names):
    """
    Analyze feature importance using multiple methods.
    """
    # Replace inf values with nan
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows with nan values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()

    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)

    # Prepare results
    results = pd.DataFrame({
        'feature': ['const'] + list(feature_names),
        'coefficient': model.params,
        'std_error': model.bse,
        't_value': model.tvalues,
        'p_value': model.pvalues,
        'mutual_info_score': np.concatenate(([0], mi_scores))
    })

    # Add significance levels
    results['significance'] = pd.cut(results['p_value'],
                                     bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
                                     labels=['***', '**', '*', 'ns'])

    return results.sort_values('p_value')

def detect_multicollinearity(X, feature_names, threshold=5):
    """
    Detect multicollinearity using VIF (Variance Inflation Factor).
    """
    # Add constant for VIF calculation
    X_with_const = sm.add_constant(X)

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = ['const'] + list(feature_names)
    vif_data["VIF"] = [variance_inflation_factor(X_with_const, i)
                       for i in range(X_with_const.shape[1])]
    vif_data['concerning_multicollinearity'] = vif_data['VIF'] > threshold

    return vif_data.sort_values('VIF', ascending=False)


def plot_correlation_heatmaps(df, output_path):
    """
    Create both full and weather-specific correlation heatmaps.
    Excludes constant and zero-variance columns.
    """
    # Remove constant and zero-variance columns
    columns_to_drop = ['const'] if 'const' in df.columns else []

    # Add min_sonnenscheinDauer to columns to drop
    if 'min_sonnenscheinDauer' in df.columns:
        columns_to_drop.append('min_sonnenscheinDauer')

    # Find and exclude zero-variance columns
    zero_var_cols = df.columns[df.var() == 0]
    columns_to_drop.extend(zero_var_cols)

    # Drop identified columns
    df_corr = df.drop(columns=columns_to_drop)

    # 1. Full correlation heatmap
    plt.figure(figsize=(24, 16))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Full Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_path}/correlation_heatmap.png")
    plt.close()

    # 2. Weather-specific heatmap
    weather_patterns = [
        'temperatur',
        'regenSchnee',
        'sonnenscheinDauer',
        'windGeschwindigkeit',
        'HDD'
    ]

    weather_cols = [col for col in df_corr.columns if any(pattern in col for pattern in weather_patterns)]

    if weather_cols:
        plt.figure(figsize=(12, 10))
        weather_corr = df_corr[weather_cols].corr()

        # Create heatmap with larger font sizes
        heatmap = sns.heatmap(weather_corr,
                              annot=True,
                              cmap='coolwarm',
                              center=0,
                              fmt='.2f',
                              square=True,
                              cbar_kws={'label': 'Correlation Coefficient'},
                              annot_kws={'size': 12})  # Increase annotation font size

        plt.title('Weather Features Correlation Heatmap', fontsize=16, pad=20)  # Larger title font
        plt.xlabel('Weather Features', fontsize=14)  # Larger x-label
        plt.ylabel('Weather Features', fontsize=14)  # Larger y-label

        # Increase tick label font sizes
        plt.xticks(fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12, rotation=0)

        # Adjust colorbar label font size
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Correlation Coefficient', fontsize=12)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(f"{output_path}/weather_correlation_heatmap.png", bbox_inches='tight', dpi=300)
        plt.close()


def plot_temporal_correlation_heatmap(df, output_path):
    """
    Create a correlation heatmap specifically for time-related features.
    """
    # Define temporal patterns to match column names
    temporal_patterns = [
        'hour_',  # For hour dummies
        'early_morning',
        'morning',
        'midday',
        'afternoon',
        'evening',
        'peak',
        'transition',
        'temp_effect'
    ]

    # Get temporal-related columns
    temporal_cols = [col for col in df.columns if any(pattern in col for pattern in temporal_patterns)]

    if not temporal_cols:
        print("No temporal-related columns found in the dataset")
        return

    # Create correlation matrix for temporal features
    temporal_corr = df[temporal_cols].corr()

    # Create heatmap
    plt.figure(figsize=(30, 24))  # Larger figure size for better readability

    # Create heatmap with larger font sizes
    heatmap = sns.heatmap(temporal_corr,
                          annot=True,
                          cmap='coolwarm',
                          center=0,
                          fmt='.2f',
                          square=True,
                          cbar_kws={'label': 'Correlation Coefficient'},
                          annot_kws={'size': 10})  # Slightly smaller font size due to more features

    plt.title('Temporal Features Correlation Heatmap', fontsize=16, pad=20)
    plt.xlabel('Temporal Features', fontsize=14)
    plt.ylabel('Temporal Features', fontsize=14)

    # Increase tick label font sizes and rotate for better readability
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10, rotation=0)

    # Adjust colorbar label font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Correlation Coefficient', fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(f"{output_path}/temporal_correlation_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()
    # Remove constant and zero-variance columns
    columns_to_drop = ['const'] if 'const' in df.columns else []

    # Add min_sonnenscheinDauer to columns to drop
    if 'min_sonnenscheinDauer' in df.columns:
        columns_to_drop.append('min_sonnenscheinDauer')

    # Find and exclude zero-variance columns
    zero_var_cols = df.columns[df.var() == 0]
    columns_to_drop.extend(zero_var_cols)

    # Drop identified columns
    df_corr = df.drop(columns=columns_to_drop)

    # 1. Full correlation heatmap
    plt.figure(figsize=(24, 16))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Full Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_path}/correlation_heatmap.png")
    plt.close()

    # 2. Weather-specific heatmap
    weather_patterns = [
        'temperatur',
        'regenSchnee',
        'sonnenscheinDauer',
        'windGeschwindigkeit',
        'HDD'
    ]

    weather_cols = [col for col in df_corr.columns if any(pattern in col for pattern in weather_patterns)]

    if weather_cols:
        plt.figure(figsize=(12, 10))
        weather_corr = df_corr[weather_cols].corr()

        # Create heatmap with larger font sizes
        heatmap = sns.heatmap(weather_corr,
                              annot=True,
                              cmap='coolwarm',
                              center=0,
                              fmt='.2f',
                              square=True,
                              cbar_kws={'label': 'Correlation Coefficient'},
                              annot_kws={'size': 12})  # Increase annotation font size

        plt.title('Weather Features Correlation Heatmap', fontsize=16, pad=20)  # Larger title font
        plt.xlabel('Weather Features', fontsize=14)  # Larger x-label
        plt.ylabel('Weather Features', fontsize=14)  # Larger y-label

        # Increase tick label font sizes
        plt.xticks(fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12, rotation=0)

        # Adjust colorbar label font size
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Correlation Coefficient', fontsize=12)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(f"{output_path}/weather_correlation_heatmap.png", bbox_inches='tight', dpi=300)
        plt.close()


def plot_feature_coefficients(importance_results, output_path, n_top=20):
    """
    Create visualization of feature coefficients and their significance.
    """
    # Remove constant before plotting
    importance_results_no_const = importance_results[importance_results['feature'] != 'const']

    plt.figure(figsize=(12, 8))

    # Sort by absolute coefficient value and get top n
    top_features = importance_results_no_const.nlargest(n_top, 'coefficient')

    # Create coefficient plot
    sns.barplot(data=top_features,
                x='coefficient',
                y='feature',
                palette=['red' if p < 0.05 else 'gray' for p in top_features['p_value']])

    plt.title(f'Top {n_top} Feature Coefficients\n(Red bars indicate statistical significance p<0.05)')
    plt.tight_layout()
    plt.savefig(f"{output_path}/top_coefficients.png")
    plt.close()


def analyze_features(df, target_col, output_path):
    """
    Comprehensive feature analysis including correlation, importance, and multicollinearity.
    Excludes constant and zero-variance columns.
    """
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Remove min_sonnenscheinDauer and zero-variance columns
    columns_to_drop = [target_col]
    if 'min_sonnenscheinDauer' in df.columns:
        columns_to_drop.append('min_sonnenscheinDauer')

    # Find and exclude zero-variance columns
    zero_var_cols = df.columns[df.var() == 0]
    columns_to_drop.extend(zero_var_cols)

    # Prepare features and target
    X = df.drop(columns=columns_to_drop)
    y = df[target_col]

    # Clean data: Remove rows with inf or nan values
    mask = ~(X.isin([np.inf, -np.inf]).any(axis=1) | X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    feature_names = X.columns

    # 1. Correlation Analysis and Heatmaps
    corr_results = calculate_correlation_matrix(df, target_col)
    corr_results.to_excel(f"{output_path}/correlation_analysis.xlsx", index=False)
    plot_correlation_heatmaps(df, output_path)
    plot_temporal_correlation_heatmap(df, output_path)

    # 2. Feature Importance Analysis
    importance_results = analyze_feature_importance(X, y, feature_names)
    importance_results.to_excel(f"{output_path}/feature_importance.xlsx", index=False)
    plot_feature_coefficients(importance_results, output_path)

    # 3. Multicollinearity Analysis
    multicollinearity_results = detect_multicollinearity(X, feature_names)
    multicollinearity_results = multicollinearity_results[multicollinearity_results['feature'] != 'const']
    multicollinearity_results.to_excel(f"{output_path}/multicollinearity.xlsx", index=False)

    # 4. Create Summary Report
    summary = pd.DataFrame({
        'metric': ['Total Features',
                   'Strong Correlations (>0.5)',
                   'Significant Features (p<0.05)',
                   'Features with Concerning Multicollinearity'],
        'value': [
            len(feature_names),
            len(corr_results[corr_results['correlation_strength'].isin(['Strong', 'Very Strong'])]),
            len(importance_results[importance_results['p_value'] < 0.05]) - 1,  # Subtract 1 for constant
            len(multicollinearity_results[multicollinearity_results['concerning_multicollinearity']])
        ]
    })
    summary.to_excel(f"{output_path}/analysis_summary.xlsx", index=False)

    return {
        'correlation': corr_results,
        'importance': importance_results,
        'multicollinearity': multicollinearity_results,
        'summary': summary
    }

# XG Boost
def analyze_xgboost_features_statistical(model, X_train, y_train, save_dir):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Statistical Analysis
    feature_stats = []
    for i, feature in enumerate(X_train.columns):
        # Calculate SHAP statistics
        shap_feature = shap_values[:, i]

        # T-test for feature significance
        t_stat, p_value = stats.ttest_1samp(shap_feature, 0)

        # Effect size (Cohen's d)
        effect_size = np.mean(shap_feature) / np.std(shap_feature)

        # Feature importance from model
        importance = model.feature_importances_[i]

        # Store results
        feature_stats.append({
            'Feature': feature,
            'SHAP_Mean': np.mean(np.abs(shap_feature)),
            'SHAP_Std': np.std(shap_feature),
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Effect_Size': effect_size,
            'Importance': importance
        })

    # Create DataFrame
    stats_df = pd.DataFrame(feature_stats)
    stats_df = stats_df.sort_values('SHAP_Mean', ascending=False)

    # Save statistical analysis
    stats_df.to_csv(f"{save_dir}/xgboost_feature_statistics.csv")

    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png")
    plt.close()

    # Feature Importance Comparison Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(stats_df['Importance'], stats_df['SHAP_Mean'])
    plt.xlabel('XGBoost Feature Importance')
    plt.ylabel('Mean |SHAP Value|')
    for i, feature in enumerate(stats_df['Feature']):
        plt.annotate(feature,
                     (stats_df['Importance'].iloc[i],
                      stats_df['SHAP_Mean'].iloc[i]))
    plt.savefig(f"{save_dir}/importance_vs_shap.png")
    plt.close()

    return stats_df
