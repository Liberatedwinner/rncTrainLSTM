import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(20201005)


class ModelCore(object):
    def __init__(self, filepath=None):
        self._filePath = filepath

    def save_data(self, filename=None, data=None):
        if self._filePath is None:
            assert filename, 'Invalid file path.'
        print('=======')
        print(f'Saving the data to {self._filePath}{filename}...')
        with open(self._filePath + filename, 'wb') as f:
            pickle.dump(data, f)
        print('The data has been saved.')
        print('=======\n')

    def load_data(self, filename=None):
        if self._filePath is None:
            assert filename, 'Invalid file path.'
        print('=======')
        print('Now loading...')
        with open(self._filePath + filename, 'rb') as f:
            data = pickle.load(f)
        print('Complete.')
        print('=======\n')
        return data

    def load_train_test_data(self):
        train_data = self.load_data('train.pkl')
        test_data = self.load_data('test.pkl')
        return train_data, test_data

    def pred_drawing(self, y_pred, y_test, picture_number, step):
        if not os.path.exists(self._filePath):
            os.makedirs(self._filePath)
        start, end = 0, len(y_test)
        plt.figure(figsize=(16, 10))
        plt.plot(y_pred[start:end], linewidth=2, linestyle="-", color="r")
        plt.plot(y_test[start:end], linewidth=2, linestyle="-", color="b")
        plt.legend(["Prediction", "Ground Truth"])
        plt.xlim(1000, 2000)
        plt.ylim(0, 120)
        plt.grid(True)
        plt.savefig(self._filePath + f'PredictedStepTest_{step}_folds_{picture_number + 1}.png',
                    dpi=50, bbox_inches="tight")
        plt.close("all")