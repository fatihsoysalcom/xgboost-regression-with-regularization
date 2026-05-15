import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate synthetic data for a regression problem
np.random.seed(42)
X = np.random.rand(100, 5) * 10 # 100 samples, 5 features
y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 2 # Target variable

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train an XGBoost Regressor model
# The article highlights Gradient Boosting and Regularization.
# XGBoost inherently uses an optimized gradient boosting framework.
# Regularization parameters (like max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda)
# are crucial for preventing overfitting and improving generalization, as discussed in the article.
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', # Objective function for regression tasks
    n_estimators=100,             # Number of boosting rounds (decision trees)
    learning_rate=0.1,            # Step size shrinkage to prevent overfitting
    max_depth=5,                  # Maximum depth of a tree, a regularization parameter
    subsample=0.8,                # Subsample ratio of the training instance, helps prevent overfitting
    colsample_bytree=0.8,         # Subsample ratio of columns when constructing each tree, helps prevent overfitting
    reg_alpha=0.1,                # L1 regularization term on weights (alpha), helps prevent overfitting
    reg_lambda=1,                 # L2 regularization term on weights (lambda), helps prevent overfitting
    random_state=42,
    n_jobs=-1                     # Use all available CPU cores
)

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)
print("Model training complete.")

# 4. Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# 5. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Demonstrate feature importance (a useful aspect of tree-based models like XGBoost)
print("\nFeature Importances:")
for i, importance in enumerate(xgb_model.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")
