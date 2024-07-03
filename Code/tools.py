import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model


def read_yearly():
    wd = Path.cwd()
    file = Path(wd, '..', 'Data', 'SN_y_tot_V2.0.csv')
    df = pd.read_csv(file, names=['year', 'mean SN',
                     'stddev', '#obs', 'prov?'], header=None, sep=';')
    return df


def make_train_sequences(data: pd.DataFrame, num_inputs: int = 22, num_preds: int = 11, data_col: str = "mean SN"):
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


def make_dense_model(cells: int = 32, num_inputs: int = 22, num_preds: int = 11):
    mod = Sequential()
    mod.add(Flatten())
    mod.add(Dense(cells))
    mod.add(Dense(num_preds))
    mod.build(input_shape=(None, num_inputs, 1))
    mod.compile(loss='mae')
    mod.summary()
    return mod


def predict(model: Model, data: pd.DataFrame, from_year: float, num_inputs: int = 22, num_preds: int = 11, year_col='year', data_col='mean SN'):
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
