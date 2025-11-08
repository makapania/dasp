import numpy as np
from sklearn.metrics import r2_score
from spectral_predict.neural_boosted import NeuralBoostedRegressor

np.random.seed(42)
n_samples=80;n_wavelengths=500
X = np.random.randn(n_samples, n_wavelengths)*0.05 + 1.0
y = 2.0 * X[:,100] + 1.5 * X[:,200] - 0.8 * X[:,300] + np.random.randn(n_samples)*0.05

# Force poor ensemble to trigger fallback
model = NeuralBoostedRegressor(n_estimators=1, learning_rate=0.1, hidden_layer_size=5, max_iter=1, early_stopping=False, random_state=42, verbose=0)
model.fit(X,y)
preds = model.predict(X)
print('r2', r2_score(y,preds))
print('fallback?', hasattr(model,'_fallback_model') and model._fallback_model is not None)