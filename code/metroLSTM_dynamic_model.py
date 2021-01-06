# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import argparse
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import ModelCheckpoint, LambdaCallback
from metroLSTM_util import *
import MetroLSTMconfig
warnings.filterwarnings('ignore')
np.random.seed(20201005)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1,
                    help='Turn GPU on(GPU number) or off(-1). Default is -1.')
parser.add_argument('--predictstep', type=int, default=10,
                    help='Choose the predicted step: for example, 1, 10, 30, 50, etc. Default value is 10.')
parser.add_argument('--activation', type=str, default='mish',
                    help='Choose the recurrent activation function: "sigmoid" or "mish". Default is mish.')
parser.add_argument('--explore_hp', type=int, default='1',
                    help='Turn the parameter search on(1) or off(0). Default is 1.')
parser.add_argument('--hs', type=int,
                    help='Determine the hidden unit size of model. This option is valid only when explore_hp is 0.')
parser.add_argument('--lr', type=float,
                    help='Determine the learning rate of model. This option is valid only when explore_hp is 0.')
parser.add_argument('--bs', type=int,
                    help='Determine the batch size of model. This option is valid only when explore_hp is 0.')
args = parser.parse_args()

predicted_step = args.predictstep
recurrent_activation = args.activation
param_search_switch = args.explore_hp
direct_input_hs = args.hs
direct_input_lr = args.lr
direct_input_bs = args.bs

if param_search_switch:
    hidden_sizes = MetroLSTMconfig.MODEL_CONFIG['hidden_sizes']
    lrs = MetroLSTMconfig.MODEL_CONFIG['learning_rates']
    batch_sizes = MetroLSTMconfig.MODEL_CONFIG['batch_sizes']
else:
    hidden_sizes = [direct_input_hs]
    lrs = [direct_input_lr]
    batch_sizes = [direct_input_bs]

os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
sns.set(style='ticks', font_scale=1.1, palette='deep', color_codes=True)

DATA_PATH = MetroLSTMconfig.MODEL_CONFIG['data_path'] + f'sec{predicted_step}//'
FILE_PATH = f'..//Plots-{recurrent_activation}//{predicted_step}//'
#######


def save_chkpt(filepath_, chkpt_):
    """
    Save the checkpoint of model to some location.
    """
    with open(filepath_ + 'chkpt_best.pkl', 'wb') as f:
        pickle.dump(chkpt_.best, f, protocol=pickle.HIGHEST_PROTOCOL)


chkpt = ModelCheckpoint(
    filepath=FILE_PATH + 'model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

save_chkpt_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: save_chkpt(FILE_PATH, chkpt)
)

#######
if __name__ == '__main__':
    trainData, testData = drop_nan_data(DATA_PATH)
    fold_number = MetroLSTMconfig.MODEL_CONFIG['fold_number']
    tscv = TimeSeriesSplit(n_splits=fold_number)
    folds = []
    for trainInd, validInd in tscv.split(trainData):
        folds.append([trainInd, validInd])
    save_chkpt_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: save_chkpt(FILE_PATH, chkpt)
    )

    # Start the time series cross validation.
    for hs in hidden_sizes:
        for lr in lrs:
            for bs in batch_sizes:
                FILE_PATH = FILE_PATH + f'{hs}-{lr}-{bs}//'
                if not os.path.exists(FILE_PATH):
                    os.makedirs(FILE_PATH)

                trained_model_score(FILE_PATH, folds, trainData, testData, hs, lr, bs)
