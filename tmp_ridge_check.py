import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

np.random.seed(42)
n_samples=80;n_wavelengths=500
X = np.random.randn(n_samples, n_wavelengths)*0.05 + 1.0
y = 2.0 * X[:,100] + 1.5 * X[:,200] - 0.8 * X[:,300] + np.random.randn(n_samples)*0.05
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
preds = ridge.predict(X_scaled)
print('ridge r2', r2_score(y,preds))