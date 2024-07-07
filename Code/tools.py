from typing import Tuple
import numpy.typing as npt
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model


def read_yearly() -> pd.DataFrame:
    """
    Reads the data file containing the annual mean sunspot numbers. Expects the proper file to be present.

    ---
    #### return: pandas.DataFrame
    Data with the columns named 'year', 'mean SN', 'stddev', '#obs', and 'prov?', respectively.
    """
    wd = Path(__file__).parent.absolute()
    file = Path(wd, '..', 'Data', 'SN_y_tot_V2.0.csv')
    df = pd.read_csv(file, names=['year', 'mean SN',
                     'stddev', '#obs', 'prov?'], header=None, sep=';')
    return df


def normalize_data(data: pd.DataFrame, data_col: str = "mean SN", norm_col: str = "norm SN") -> Tuple[float, float]:
    """
    Computes a normalized version of the data column in the data frame by subtracting the mean and dividing
    by the standard deviation. These normalized data are appended to tha data frame as a new column.

    ---
    #### data: pandas.DataFrame
    The data frame with all the data

    #### data_col: str (default: "mean SN")
    The column with the data that is normalized.

    #### norm_col: str (default: "norm SN")
    Name for the new column with the normalized data.

    ---
    #### return: (float, float)
    Computed mean in the first position and computed standard deviation in the second position.
    """
    sn = data[data_col]
    mean = pd.Series.mean(sn)
    stddev = pd.Series.std(sn)
    data[norm_col] = (sn-mean)/stddev
    return mean, stddev


def denormalize_data(data: pd.Series, mean: float, stddev: float) -> pd.Series:
    """
    Denormalizes a series by multiplying with the provided 'stddev' first and adding 'mean' afterwards.

    ---
    #### data: pandas.Series
    The data series that is denormalized.

    #### mean: float
    The mean that is added to the data.

    #### stddev: float
    The standard deviation the data is multiplied with.

    ---
    #### return: pd.Series
    The denormalized data.
    """
    return data*stddev+mean


def make_train_sequences(data: pd.DataFrame, num_inputs: int = 22, num_preds: int = 11, data_col: str = "mean SN") -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes training samples. Each sample is a sequence of 'num_inputs' consecutive data points used as features and
    the following 'num_preds' data points that serve as labels. Each possible sequence extractable from the given data
    is used. Note that you still have to split the sequences into training data, test data, and validation data.

    ---
    #### data: pd.DataFrame
    All of our data.

    #### num_inputs: int (default: 22)
    Number of consecutive data points that serve as features.

    #### num_preds: int (default: 11)
    Number of consecutive data points that serve as labels.

    #### data_col: str (default: "mean SN")
    Name of the column containing the data.

    ---
    #### return: (np.ndarray, np.ndarray)
    First array are the features, second are the labels. If N sequences were generated, the
    shape of the first array is (N, num_inputs) and the shape of the second array is (N, num_preds).
    The i-th entry of the first array and the i-th entry of the second array belong together.
    """
    vals = data[data_col]
    startIndices = np.arange(vals.shape[0]-num_preds-num_inputs+1)
    inputs = np.array(vals[startIndices[0]:startIndices[0]+num_inputs])
    labels = np.array(
        vals[startIndices[0]+num_inputs:startIndices[0]+num_inputs+num_preds])
    for idx in startIndices[1:]:
        # num_inputs consecutive values from vals, starting at idx
        arr = np.array(vals[idx:idx+num_inputs])
        inputs = np.vstack([inputs, arr])
        # num_preds consecutive values fom vals, just following the values used for the inputs
        arr = np.array(vals[idx+num_inputs:idx+num_inputs+num_preds])
        labels = np.vstack([labels, arr])

    return inputs, labels


def make_dense_model(cells: int = 32, num_inputs: int = 22, num_preds: int = 11) -> Model:
    """
    Creates a simple neural network with one hidden layer of 'cells' densly connected neurons and 'num_preds' output neurons.

    ---
    #### cells: int (default: 32)
    Number of cells in the hidden layer.

    #### num_inputs: int (default: 22)
    Number of input values.

    #### num_preds: int (default: 11)
    Number of output values.

    ---

    #### return: tensorflow.keras.Model
    The untrained neural network.
    """
    mod = Sequential()
    mod.add(Flatten())
    mod.add(Dense(cells))
    mod.add(Dense(num_preds))
    mod.build(input_shape=(None, num_inputs, 1))
    mod.compile(loss='mae')
    mod.summary()
    return mod


def make_lstm_model(cells: int = 32, num_inputs: int = 22, num_preds: int = 11) -> Model:
    """
    Creates a simple neural network with one hidden layer of 'cells' LSTM neurons and 'num_preds' output neurons.

    ---
    #### cells: int (default: 32)
    Number of LSTMs in the hidden layer.

    #### num_inputs: int (default: 22)
    Number of input values.

    #### num_preds: int (default: 11)
    Number of output values.

    ---
    #### return: tensorflow.keras.Model
    The untrained neural network.
    """
    mod = Sequential()
    mod.add(LSTM(cells))
    mod.add(Dense(num_preds))
    mod.build(input_shape=(None, num_inputs, 1))
    mod.compile(loss='mae')
    mod.summary()
    return mod


def predict(model: Model, data: pd.DataFrame, from_year: float, num_inputs: int = 22, num_preds: int = 11, year_col='year', data_col='mean SN') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Makes a prediction.

    ---
    #### model: tensorflow.keras.Model

    The model that predicts data from the inputs.

    #### data: pandas.DataFrame

    All of our data.

    #### from_year: float

    A year from which on the prediction is made. The integer numbers correspond to new year. Use fractionals for specifiying
    starting points within a year.

    #### num_inputs: int (default: 22)

    Number of consecutive data points that are taken as input. The last data point in the input sequence is the best match to
    the year provided.

    #### num_preds: int (deafult: 11)

    Number of data points that are predicted. These data points immediately follow the time stamp provided in 'from_year'.

    #### year_col: str (default: "year")

    Name of the column in the data frame containign the years. The argument 'from_year' is compared against these numbers for
    determining the point where the prediction starts.

    #### data_col: str (default: "mean SN")

    Name of the column in the data frame containing the actual data that is used as input.

    ---
    #### return: np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray

    Arrays in this order: input dates, input values, output dates, output values, label dates, label values

    The inputs are used as inputs for the prediction. The output values are the predictions made. Id you predict
    data for a time interval where actual data is provided, these actual data is returned as labels. This
    allows for an easy comparison of the prediction with the actual data. If you really predict into the future
    only (i.e. using the very last 'num_inputs' data points as inputs), label dates and label values are 'None'.
    """
    years = np.array(data[year_col])
    pred_from_idx = np.argmin(np.abs(years-from_year)) + 1
    input_from_idx = pred_from_idx - num_inputs
    if input_from_idx < 0:
        raise ValueError(
            "Not enough input values. Consider a later year to predict from.")

    allvals = np.array(data[data_col])
    if pred_from_idx < allvals.size:
        # predicting data of the past, so get these true values
        labels_to_idx = np.min([allvals.size, pred_from_idx+num_preds])
        labels = allvals[pred_from_idx-1: labels_to_idx]
        label_dates = years[pred_from_idx-1: labels_to_idx]
    else:
        labels = None
        label_dates = None

    input_dates = years[input_from_idx:pred_from_idx]
    invals = allvals[input_from_idx:pred_from_idx]

    output_dates = np.arange(num_preds)+1
    output_dates = output_dates*(years[years.size-1]-years[years.size-2])
    output_dates = output_dates + years[pred_from_idx-1]
    outputs = model.predict(invals.reshape(1, num_inputs))

    return input_dates, invals, output_dates, outputs[0], label_dates, labels
