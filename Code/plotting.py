import matplotlib.pyplot as plt
import numpy as np
import tools as tl
import analytical as an
import pmdarima as pm
from pmdarima.preprocessing import BoxCoxEndogTransformer as BC
from pmdarima.pipeline import Pipeline


def plot_annual_example() -> None:
    num_inputs = 22
    num_preds = 11
    hidden_cells = 32

    df = tl.read_yearly()
    mean, stddev = tl.normalize_data(df)

    endOf22 = 1996.5
    endOf23 = 2008.5
    firstOf23 = 1 + np.argmin(np.abs(df['year']-endOf22))
    firstOf24 = 1 + np.argmin(np.abs(df['year']-endOf23))

    # training data
    inputs, labels = tl.make_train_sequences(
        df[:firstOf23], num_inputs=num_inputs, num_preds=num_preds, data_col="norm SN")
    splitAt = int(np.ceil(inputs.shape[0] * 0.8))
    train_inputs = inputs[:splitAt]
    train_labels = labels[:splitAt]
    val_inputs = inputs[splitAt:]
    val_labels = labels[splitAt:]

    # dense model
    denseModel = tl.make_dense_model(
        cells=hidden_cells, num_inputs=num_inputs, num_preds=num_preds)
    denseModel.fit(train_inputs, train_labels, batch_size=8,
                   epochs=20, validation_data=(val_inputs, val_labels))
    inputDates23, inVals23, outputDates23, outDense23, labelDates23, labels23 = tl.predict(
        denseModel, df, from_year=endOf22, num_inputs=num_inputs, num_preds=num_preds, data_col="norm SN")
    outDense23 = tl.denormalize_data(outDense23, mean=mean, stddev=stddev)

    inputDates24, inVals24, outputDates24, outDense24, labelDates24, labels24 = tl.predict(
        denseModel, df, from_year=endOf23, num_inputs=num_inputs, num_preds=num_preds, data_col="norm SN")
    outDense24 = tl.denormalize_data(outDense24, mean=mean, stddev=stddev)

    # lstm model
    lstmModel = tl.make_dense_model(
        cells=hidden_cells, num_inputs=num_inputs, num_preds=num_preds)
    lstmModel.fit(train_inputs, train_labels, batch_size=8,
                  epochs=20, validation_data=(val_inputs, val_labels))
    inputDates23, inVals23, outputDates23, outLstm23, labelDates23, labels23 = tl.predict(
        lstmModel, df, from_year=endOf22, num_inputs=num_inputs, num_preds=num_preds, data_col="norm SN")
    outLstm23 = tl.denormalize_data(outLstm23, mean=mean, stddev=stddev)

    inputDates24, inVals24, outputDates24, outLstm24, labelDates24, labels24 = tl.predict(
        lstmModel, df, from_year=endOf23, num_inputs=num_inputs, num_preds=num_preds, data_col="norm SN")
    outLstm24 = tl.denormalize_data(outLstm24, mean=mean, stddev=stddev)

    # extrapolate from values exactly one and two periods ago
    outputDates23, outExtrapol23 = an.extrapolate(
        df, from_year=endOf22, period=num_preds, data_col="mean SN")
    outputDates24, outExtrapol24 = an.extrapolate(
        df, from_year=endOf23, period=num_preds, data_col="mean SN")

    # arima
    arima = Pipeline([('boxcox', BC(lmbda2=1e-6)), ('auto_arima',
                     pm.AutoARIMA(m=num_preds, seasonal=True, surpress_warnings=True))])
    arima.fit(df.iloc[:firstOf23]["mean SN"])
    outArima23 = arima.predict(n_periods=num_preds)

    arima.update(df.iloc[firstOf23:firstOf24]["mean SN"])
    outArima24 = arima.predict(n_periods=num_preds)

    # denormalize
    inVals23 = tl.denormalize_data(inVals23, mean=mean, stddev=stddev)
    inVals24 = tl.denormalize_data(inVals24, mean=mean, stddev=stddev)
    labels23 = tl.denormalize_data(labels23, mean=mean, stddev=stddev)
    labels24 = tl.denormalize_data(labels24, mean=mean, stddev=stddev)

    # prepare plot
    fig = plt.figure()

    ax1 = fig.add_axes([.1, .6, .85, .4])
    ax1.set(xlim=(1975, 2010), ylim=(-10, 230), ylabel="mean sunspot number")
    ax1.text(1980.8, 20, "21", ha="center")
    ax1.text(1991.0, 20, "22", ha="center")
    ax1.text(2001.4, 20, "23", ha="center")

    ax2 = fig.add_axes([.1, .15, .85, .4])
    ax2.set(xlim=(1985, 2020), ylim=(-10, 230),
            xlabel="year", ylabel="mean sunspot number")
    ax2.text(1991.0, 20, "22", ha="center")
    ax2.text(2001.4, 20, "23", ha="center")
    ax2.text(2014.0, 20, "24", ha="center")

    # plot predictions for cycle 23
    ax1.plot(inputDates23, inVals23, color="black",
             linestyle="-", label="previous cycles")
    ax1.plot(labelDates23, labels23, color="black",
             linestyle="--", label="predicted cycle")
    ax1.plot(outputDates23, outExtrapol23, color="green",
             marker="d", markerfacecolor="white", linestyle="None", label="extrapolation")
    ax1.plot(outputDates23, outArima23, color="orange",
             marker="o", markerfacecolor="white", linestyle="None", label="SARIMAX")
    ax1.plot(outputDates23, outDense23, color="blue",
             marker="+", linestyle="None", label="dense NN")
    ax1.plot(outputDates23, outLstm23, color="red",
             marker="x", linestyle="None", label="LSTM")

    # plot predictions for cycle 24
    ax2.plot(inputDates24, inVals24, color="black", linestyle="-")
    ax2.plot(labelDates24, labels24, color="black", linestyle="--")
    ax2.plot(outputDates24, outExtrapol24, color="green",
             marker="d", markerfacecolor="white", linestyle="None")
    ax2.plot(outputDates24, outArima24, color="orange",
             marker="o", markerfacecolor="white", linestyle="None")
    ax2.plot(outputDates24, outDense24, color="blue",
             marker="+", linestyle="None")
    ax2.plot(outputDates24, outLstm24, color="red",
             marker="x", linestyle="None")

    fig.legend(loc='lower center', ncol=3)
    plt.show()
