import numpy as np
from sklearn.metrics import r2_score
from spectral_predict.neural_boosted import NeuralBoostedRegressor

np.random.seed(42)
n_samples=50;n_wavelengths=2000
X = np.random.randn(n_samples, n_wavelengths)*0.1 + 1.0
y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples)*0.1
model = NeuralBoostedRegressor(n_estimators=20, learning_rate=0.1, hidden_layer_size=3, early_stopping=True, random_state=42, verbose=1)
model.fit(X,y)
preds = model.predict(X)
print('r2', r2_score(y,preds))
print('fallback?', hasattr(model,'_fallback_model') and model._fallback_model is not None)