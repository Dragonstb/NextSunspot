import numpy as np
import numpy.typing as npt
import pandas as pd


def extrapolate(data: pd.DataFrame, from_year: float, period: int = 11, year_col="year", data_col="mean SN") -> npt.NDArray:
    """
    Simply forecasts a value by linearly extrapolating from the values one and two periods before.

    #### data: pandas.DataFrame
    All of our data.

    #### from_year: float
    A year from which on the prediction is made. The integer numbers correspond to new year. Use fractionals for specifiying
    starting points within a year.

    #### period: float (default: 11)
    Number of data points within a solar cycle.

    #### year_col: str (default: "year")
    Name of the column in the data frame containign the years. The argument 'from_year' is compared against these numbers for
    determining the point where the prediction starts.

    #### data_col: str (default: "mean SN")
    Name of the column in the data frame containing the actual data that is used as input.

    ---
    return: np.NDArray
    extrapolated values. The length of the array matches the number specified in 'period'
    """
    if period < 1:
        raise ValueError('period must be positive.')

    years = np.array(data[year_col])
    pred_from_index = np.argmin(np.abs(years-from_year)) + 1
    pred_to_index = pred_from_index + period
    one_period_ago = np.array(
        data.iloc[pred_from_index-period:pred_to_index-period][data_col])
    two_periods_ago = np.array(
        data.iloc[pred_from_index-2*period:pred_to_index-2*period][data_col])
    extr = 2*one_period_ago - two_periods_ago
    return extr
