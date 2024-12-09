import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('Housing_Prices_India.csv')

df.replace(9, np.nan, inplace=True)  

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])  

label_encoders = {}
categorical_columns = ['Location']  
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

amenity_columns = ['MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack']
for col in amenity_columns:
    df[col] = df[col].map({0: 0, 1: 1, 9: np.nan})  

X = df.drop(columns=['Price'])  
y = df['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_model.fit(X_train, y_train)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters for Gradient Boosting:", grid_search.best_params_)
gbr_model = grid_search.best_estimator_

lr_cv_score = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
gbr_cv_score = cross_val_score(gbr_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

print(f"Linear Regression CV MSE: {-lr_cv_score.mean()}")
print(f"Gradient Boosting Regressor CV MSE: {-gbr_cv_score.mean()}")

lr_predictions = lr_model.predict(X_test)
gbr_predictions = gbr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

gbr_mse = mean_squared_error(y_test, gbr_predictions)
gbr_r2 = r2_score(y_test, gbr_predictions)

print("\nLinear Regression:")
print("MSE:", lr_mse)
print("R-squared:", lr_r2)

print("\nGradient Boosting Regressor:")
print("MSE:", gbr_mse)
print("R-squared:", gbr_r2)

feature_importance = gbr_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Gradient Boosting Regressor:")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_predictions, label='Linear Regression Predictions', alpha=0.6)
plt.scatter(y_test, gbr_predictions, label='Gradient Boosting Predictions', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()


df[numerical_columns].hist(bins=20, figsize=(15, 10), layout=(8, 5))
plt.suptitle('Feature Distributions (Histograms)')
plt.tight_layout()  
plt.show()

corr = df[numerical_columns].corr()
plt.figure(figsize=(40, 20))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 12})
plt.title('Correlation Heatmap')
plt.show()

lr_residuals = y_test - lr_predictions
gbr_residuals = y_test - gbr_predictions

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(lr_residuals, bins=20, kde=True, color='blue')
plt.title('Linear Regression Residuals')

plt.subplot(1, 2, 2)
sns.histplot(gbr_residuals, bins=20, kde=True, color='green')
plt.title('Gradient Boosting Residuals')

plt.tight_layout()
plt.show()

models = ['Linear Regression', 'Gradient Boosting']
mse_values = [lr_mse, gbr_mse]
r2_values = [lr_r2, gbr_r2]

plt.figure(figsize=(10, 5))
sns.barplot(x=models, y=mse_values, palette='Blues')
plt.title('Model Comparison - MSE')
plt.ylabel('Mean Squared Error (MSE)')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=models, y=r2_values, palette='Blues')
plt.title('Model Comparison - R-squared')
plt.ylabel('R-squared')
plt.show()

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(y_test, bins=20, kde=True, color='blue', label='Actual Prices')
sns.histplot(lr_predictions, bins=20, kde=True, color='orange', label='Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(y_test, bins=20, kde=True, color='blue', label='Actual Prices')
sns.histplot(gbr_predictions, bins=20, kde=True, color='green', label='Predicted Prices')
plt.title('Gradient Boosting: Actual vs Predicted Prices')
plt.legend()

plt.tight_layout()
plt.show()

k_values = range(1, 21)
knn_mse_values = []

for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    knn_mse = mean_squared_error(y_test, knn_predictions)
    knn_mse_values.append(knn_mse)

plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_mse_values, marker='o', color='red')
plt.title('KNN Regressor Performance vs k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.show()
