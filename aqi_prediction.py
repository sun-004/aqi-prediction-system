"""
Air Quality Index (AQI) Prediction using Traffic & Weather Data
Complete ML Pipeline with Random Forest and XGBoost

Save this as: aqi_prediction.py
Run with: python aqi_prediction.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
import joblib
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class AQIPredictor:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate realistic synthetic data for AQI prediction"""
        print("📊 Generating synthetic data...")
        
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        hour = dates.hour
        day_of_week = dates.dayofweek
        month = dates.month
        
        # Weather data with seasonal patterns
        temperature = 20 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 3, n_samples)
        humidity = 60 + 15 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 10, n_samples)
        humidity = np.clip(humidity, 20, 100)
        wind_speed = np.abs(np.random.gamma(2, 2, n_samples))
        rainfall = np.random.exponential(0.5, n_samples) * (np.random.random(n_samples) < 0.15)
        
        # Traffic data with peak hours pattern
        base_traffic = 100
        peak_morning = (hour >= 7) & (hour <= 9)
        peak_evening = (hour >= 17) & (hour <= 19)
        weekend = (day_of_week >= 5)
        
        vehicle_count = base_traffic + 150 * peak_morning + 180 * peak_evening
        vehicle_count = vehicle_count * (0.6 if weekend.any() else 1.0)
        vehicle_count += np.random.normal(0, 30, n_samples)
        vehicle_count = np.clip(vehicle_count, 20, 400)
        
        congestion_level = (vehicle_count / 400) * 10
        congestion_level += np.random.normal(0, 1, n_samples)
        congestion_level = np.clip(congestion_level, 0, 10)
        
        avg_speed = 60 - (congestion_level * 4) + np.random.normal(0, 5, n_samples)
        avg_speed = np.clip(avg_speed, 10, 80)
        
        # Pollution parameters influenced by traffic and weather
        pm25_base = 30 + (vehicle_count * 0.15) - (wind_speed * 3) + (humidity * 0.1)
        pm25_base -= (rainfall * 10)
        pm25 = pm25_base + np.random.normal(0, 10, n_samples)
        pm25 = np.clip(pm25, 5, 250)
        
        pm10 = pm25 * 1.5 + np.random.normal(0, 15, n_samples)
        pm10 = np.clip(pm10, 10, 400)
        
        no2 = 20 + (vehicle_count * 0.1) + np.random.normal(0, 5, n_samples)
        no2 = np.clip(no2, 5, 150)
        
        co = 0.5 + (vehicle_count * 0.002) + np.random.normal(0, 0.2, n_samples)
        co = np.clip(co, 0.1, 5)
        
        o3 = 40 + (temperature * 0.8) - (humidity * 0.2) + np.random.normal(0, 8, n_samples)
        o3 = np.clip(o3, 10, 120)
        
        so2 = 10 + np.random.normal(0, 3, n_samples)
        so2 = np.clip(so2, 2, 50)
        
        # Calculate AQI
        aqi = self.calculate_aqi(pm25, pm10, no2, co, o3, so2)
        
        data = pd.DataFrame({
            'timestamp': dates, 'pm25': pm25, 'pm10': pm10, 'no2': no2,
            'co': co, 'o3': o3, 'so2': so2, 'vehicle_count': vehicle_count,
            'congestion_level': congestion_level, 'avg_speed': avg_speed,
            'temperature': temperature, 'humidity': humidity,
            'wind_speed': wind_speed, 'rainfall': rainfall, 'aqi': aqi
        })
        
        print(f"✅ Generated {len(data)} samples")
        return data
    
    def calculate_aqi(self, pm25, pm10, no2, co, o3, so2):
        """Simplified AQI calculation"""
        aqi = (pm25 * 2.0 + pm10 * 0.5 + no2 * 0.3 + 
               co * 10 + o3 * 0.4 + so2 * 0.5)
        return np.clip(aqi, 0, 500)
    
    def preprocess_data(self, data):
        """Feature engineering and preprocessing"""
        print("\n🔧 Preprocessing data...")
        df = data.copy()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                               (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # Rolling averages
        df['pm25_rolling_3h'] = df['pm25'].rolling(window=3, min_periods=1).mean()
        df['pm25_rolling_24h'] = df['pm25'].rolling(window=24, min_periods=1).mean()
        df['traffic_rolling_3h'] = df['vehicle_count'].rolling(window=3, min_periods=1).mean()
        
        # Interaction features
        df['traffic_wind_interaction'] = df['vehicle_count'] * (1 / (df['wind_speed'] + 1))
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        
        # Lag features
        df['pm25_lag1'] = df['pm25'].shift(1)
        df['traffic_lag1'] = df['vehicle_count'].shift(1)
        
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✅ Preprocessed data shape: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """Prepare features and target for modeling"""
        feature_cols = [
            'pm25', 'pm10', 'no2', 'co', 'o3', 'so2',
            'vehicle_count', 'congestion_level', 'avg_speed',
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
            'pm25_rolling_3h', 'pm25_rolling_24h', 'traffic_rolling_3h',
            'traffic_wind_interaction', 'temp_humidity_interaction',
            'pm25_lag1', 'traffic_lag1'
        ]
        
        X = df[feature_cols]
        y = df['aqi']
        self.feature_names = feature_cols
        
        return X, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train Random Forest and XGBoost models"""
        print("\n🤖 Training models...")
        
        print("\n1️⃣ Training Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf_base, rf_params, cv=3, 
                               scoring='neg_mean_squared_error', 
                               verbose=1, n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        self.rf_model = rf_grid.best_estimator_
        print(f"✅ Best RF params: {rf_grid.best_params_}")
        
        print("\n2️⃣ Training XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
        xgb_grid = GridSearchCV(xgb_base, xgb_params, cv=3,
                                scoring='neg_mean_squared_error',
                                verbose=1, n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        
        self.xgb_model = xgb_grid.best_estimator_
        print(f"✅ Best XGB params: {xgb_grid.best_params_}")
        
        return self.rf_model, self.xgb_model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate models and display metrics"""
        print("\n📈 Model Evaluation:")
        print("="*60)
        
        models = {'Random Forest': self.rf_model, 'XGBoost': self.xgb_model}
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'predictions': y_pred, 'rmse': rmse, 'mae': mae, 'r2': r2
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  R²:   {r2:.4f}")
        
        print("="*60)
        return results
    
    def plot_results(self, X_test, y_test, results):
        """Create visualization plots"""
        print("\n📊 Creating visualizations...")
        os.makedirs('output', exist_ok=True)
        
        # Actual vs Predicted
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for idx, (name, result) in enumerate(results.items()):
            axes[idx].scatter(y_test, result['predictions'], alpha=0.5, s=10)
            axes[idx].plot([y_test.min(), y_test.max()], 
                          [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[idx].set_xlabel('Actual AQI')
            axes[idx].set_ylabel('Predicted AQI')
            axes[idx].set_title(f'{name}\nR² = {result["r2"]:.4f}')
            axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: output/actual_vs_predicted.png")
        
        # Feature Importance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for idx, (name, model) in enumerate([('Random Forest', self.rf_model), 
                                             ('XGBoost', self.xgb_model)]):
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                axes[idx].barh(importance['feature'], importance['importance'])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{name} - Top 15 Features')
                axes[idx].invert_yaxis()
        plt.tight_layout()
        plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: output/feature_importance.png")
        
        plt.close('all')
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving models...")
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.rf_model, 'models/random_forest_model.pkl')
        joblib.dump(self.xgb_model, 'models/xgboost_model.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        print("✅ Models saved in 'models/' directory")
    
    def predict_aqi_category(self, aqi_value):
        """Categorize AQI value"""
        if aqi_value <= 50:
            return "Good", "🟢"
        elif aqi_value <= 100:
            return "Moderate", "🟡"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups", "🟠"
        elif aqi_value <= 200:
            return "Unhealthy", "🔴"
        elif aqi_value <= 300:
            return "Very Unhealthy", "🟣"
        else:
            return "Hazardous", "🟤"


def main():
    """Main execution pipeline"""
    print("="*60)
    print("  AIR QUALITY INDEX (AQI) PREDICTION SYSTEM")
    print("="*60)
    
    predictor = AQIPredictor()
    
    # Generate data
    data = predictor.generate_synthetic_data(n_samples=10000)
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/aqi_data.csv', index=False)
    print(f"✅ Raw data saved to 'data/aqi_data.csv'")
    
    # Preprocess
    processed_data = predictor.preprocess_data(data)
    
    # Prepare features
    X, y = predictor.prepare_features(processed_data)
    
    # Train-test split
    print("\n✂️ Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train models
    predictor.train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate
    results = predictor.evaluate_models(X_test, y_test)
    
    # Visualize
    predictor.plot_results(X_test, y_test, results)
    
    # Save models
    predictor.save_models()
    
    # Sample predictions
    print("\n🔮 Sample Predictions:")
    print("="*60)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for idx in sample_indices:
        actual = y_test.iloc[idx]
        rf_pred = predictor.rf_model.predict(X_test.iloc[idx:idx+1])[0]
        xgb_pred = predictor.xgb_model.predict(X_test.iloc[idx:idx+1])[0]
        
        actual_cat, actual_emoji = predictor.predict_aqi_category(actual)
        rf_cat, rf_emoji = predictor.predict_aqi_category(rf_pred)
        xgb_cat, xgb_emoji = predictor.predict_aqi_category(xgb_pred)
        
        print(f"\nSample {idx}:")
        print(f"  Actual AQI: {actual:.1f} {actual_emoji} ({actual_cat})")
        print(f"  RF Predicted: {rf_pred:.1f} {rf_emoji} ({rf_cat})")
        print(f"  XGB Predicted: {xgb_pred:.1f} {xgb_emoji} ({xgb_cat})")
    
    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print("📁 Check 'output/' folder for visualizations")
    print("📁 Check 'models/' folder for saved models")
    print("📁 Check 'data/' folder for generated dataset")
    print("="*60)


if __name__ == "__main__":
    main()
    
