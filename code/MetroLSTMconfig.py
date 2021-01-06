from keras.callbacks import EarlyStopping

MODEL_CONFIG = {
    'early_stopping': EarlyStopping(monitor='val_loss', patience=10, verbose=2),
    'data_path': f'..//Data//TrainedRes//',
    'fold_number': 10,
    'epochs': 500,
    'hidden_sizes': [10, 14, 18, 22, 26, 30],
    'learning_rates': [1e-5, 1e-4, 2e-4, 5e-4],
    'batch_sizes': [32, 64, 128, 256, 512]
}
# 26, 1e-4, 32
