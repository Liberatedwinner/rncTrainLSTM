import argparse

parser = argparse.ArgumentParser()
parser.add_argument('activation1', default='tanh',
                    help='choose the activation function instead of tanh: swish, mish')
parser.add_argument('activation2', default='sigmoid',
                    help='choose the activation function instead of sigmoid: swish, mish')
args = parser.parse_args()

###########################################
# class activations(tf.keras):
#     def __init__(self, x):
#         self.x = x
#
#     def swish(x):
#         return x * tf.nn.sigmoid(x)
#     # 이건 만들어져있으니까 따로 할 필욘 없음
#
#     def mish(x):
#         return x * tf.nn.tanh(tf.nn.softplus(x))
#     # 이건 애드온으로 있는데...
#
import tensorflow as tf
import tensorflow_addons as tfa # pip install tensorflow-addons

swish = tf.keras.activations.swish
mish = tfa.activations.mish

tf.keras.layers.LSTM(
    units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False,
    return_state=False, go_backwards=False, stateful=False, time_major=False,
    unroll=False, **kwargs
)
##########################################


x = 0
activation1, activation2 = tanh, sigmoid
# in activation function module...
if args.activation1 == 'swish':
    activation1 = swish
elif args.activation1 == 'mish':
    activation1 = mish
else:
    activation1 = 'tanh'

if args.activation2 == 'swish':
    activation2 = swish
elif args.activation2 == 'mish':
    activation2 = mish
else:
    activation2 = 'sigmoid'

tf.keras.layers.LSTM = tf.keras.layers.LSTM(activation=activation1,
                                            recurrent_activation=activation2)
###########################################



optimizers = ['adam', 'radam']
hiddensizes = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
batches = [32, 64, 128, 256, 512, 1024]
lrs = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
hiddenlayers = [1, 2, 3]



param_grid = {
    'lr': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    'optimizer': ['adam', 'radam']
}

model = KerasRegressor(build_fn = build_cnn, verbose=0)
model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model,
                                        param_grid, cv=5, scoring_fit='neg_log_loss')

print(model.best_score_)
print(model.best_params_)








chkpt = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.4f}.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)

if os.path.exists('chkpt_best.pkl'):
  with open('chkpt_best.pkl', 'rb') as f:
    best = pickle.load(f)
    chkpt.best = best

save_chkpt_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: save_chkpt()
)