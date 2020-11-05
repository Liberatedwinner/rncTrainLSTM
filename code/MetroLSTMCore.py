import numpy as np
import pickle

np.random.seed(20201005)


class SaveNLoad(object):
    def __init__(self, filename=None, filepath=None):
        self._fileName = filename
        self._filePath = filepath

    def save_data(self, data=None, path=None):
        if path is None:
            assert self._fileName, 'Invalid file path.'
            self.__save_data(data)
        else:
            self._fileName = path
            self.__save_data(data)

    def load_data(self, path=None):
        if path is None:
            assert self._fileName, 'Invalid file path.'
            return self.__load_data()
        else:
            self._fileName = path
            return self.__load_data()

    def __save_data(self, data=None):
        assert self._fileName, 'Invalid file name.'
        print('=======')
        print(f'Saving the data to a filename {self._fileName}...')
        with open(self._fileName, 'wb') as f:
            pickle.dump(data, f)
        print('The data has been saved.')
        print('=======\n')

    def __load_data(self):
        assert self._fileName, 'Invalid file name.'
        print('=======')
        print('Now loading...')
        with open(self._fileName, 'rb') as f:
            data = pickle.load(f)
        print('Complete.')
        print('=======\n')
        return data

    def load_train_test_data(self):
        train_data = self.load_data(self._filePath + 'train.pkl')
        test_data = self.load_data(self._filePath + 'test.pkl')
        return train_data, test_data
