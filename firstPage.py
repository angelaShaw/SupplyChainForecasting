import psycopg2
from flask import render_template, Flask, request, url_for, session,jsonify, sessions,send_from_directory
import datetime as dt
import json
from pandas import DataFrame
from datetime import datetime
import os
import xlrd
from openpyxl.workbook import Workbook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
pd.options.mode.chained_assignment = None
import time
import matplotlib.style as style
import warnings
import xgboost as xgb


app = Flask(__name__)
app.secret_key = "Test_Secret_Key"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
@app.route('/')
def page():
    return render_template('page1.html')

@app.route('/singleDataPointSubmission', methods=['POST', 'GET'])
def singlePage():
    material = 'x'
    group = 'y'
    date = 'z'
    if request.method == "POST":
        material= int(request.form['material'])
        group = int(request.form['group'])
        eduration = int(request.form['eduration']) #expected duration
        order_quantity = int(request.form['order_quantity'])
        cdate = request.form['cdate']
        unit = request.form['unit']
        mcat = request.form['material_category'] #material category
        newDf = single_reader(group, material, cdate, unit, order_quantity, mcat, eduration)
        #call sing forecasting method
        return render_template('downloadExcel.html',file_name = 'forecast.xlsx',outTable=[newDf.to_html()]) #figure out how to upload excel
    return render_template('singlePage.html')


@app.route('/ExcelSheetSubmission', methods=['POST', 'GET'])
def excelSub():
    target = os.path.join(APP_ROOT, 'store')
    if request.method == "POST":
        #print(request.files)
        f = request.files['file']
        filename=f.filename
        destination = "/".join([target, filename])
        f.save(destination)
        newDf = excel_reader(f)
        return render_template('downloadExcel.html', file_name='newExcel.xlsx', outTable=[newDf.to_html()])
    return render_template('excelSub.html')

def excel_reader(x):
    df = pd.read_excel(x)
    df = df.reset_index(drop=False)
    df[['duration','lower error','upper error']] = df.apply(lambda row: forecast(pd.DataFrame(row).transpose()), axis=1)
    #df['duration'] = pd.to_numeric(df['duration'], downcast='integer')
    #df = df.reset_index(drop=False)
    #df['end date'] = df.apply(lambda row: pd.bdate_range(row['Confirmation Start Date'], periods = int(row['duration'])[-1], axis=1))
    df.to_excel('store/newExcel.xlsx')
    return df

def single_reader(group, material, confirmed_start_date, unit, toq, mcat, eduration):
    csd = datetime.strptime(confirmed_start_date, '%Y/%m/%d')
    # ssd = datetime.strptime(scheduled_start_date, '%Y/%m/%d')
    # sfd = datetime.strptime(scheduled_finish_date, '%Y/%m/%d')
    cols = {'Confirmation Start Date': csd,'Group':group, 'MaterialCategory' : mcat, 'Total order quantity': int(toq), 'scheduled_duration_nowkend': eduration, 'Unit_G': int(unit=='G'), 'Unit_KU': int(unit=='KU'), 'Unit_ML': int(unit=='ML'), 'Unit_PC': int(unit=='PC'),'Unit_µMO': int(unit=='Unit_µMO'), 'Unit_µg': int(unit=='Unit_µg')  }
    # cols = {'Confirmation Start Date': csd, 'Schedule Start Date': ssd, 'Schedule Finish Date': sfd, 'Group':group, 'Material': material, 'Unit': unit, 'Total order quantity': toq, 'MaterialCategory' : mcat}
    df = pd.DataFrame()
    df = df.append(cols, ignore_index=True)
    #print(df.head(5))
    #df = DataFrame(list(cols.items()), columns=['confirmed start date', 'scheduled start date','scheduled finish date', 'group','material', 'unit', 'total order quantity' ])
    x = forecast(df).tolist()
    df['duration'] = x[0]
    df['lower error'] = x[1]
    df['upper error'] = x[2]
    end_date = pd.bdate_range(csd, periods = int(df['duration']))[-1]
    # vals = []
    # vals.append(end_date)
    df['end date']= end_date
    df = df.drop('actual_duration_nowkend', 1)
    #cols2 = {'duration':df['duration'], 'forecasted end date':end_date, 'lower error':df['lower error'], 'upper error':df['upper error']}
    #df = df.append(cols2, ignore_index=True)
    #print(df.head(5))



    ###storing df in an excel file
    df.to_excel('store/forecast.xlsx')
    # filename = f.filename
    # target = os.path.join(APP_ROOT, 'store')
    # destination = "/".join([target, filename])
    # f.save(destination)
    filename = 'forecast.xlsx'
    return df



@app.route('/store/<filename>')
def send_image(filename):
    return send_from_directory("store", filename)


def forecast(datapoint):
    # Reading DataPoint to be forecasted
    df = datapoint
    x = int(df.iloc[0]["Group"])
    group_number = str(x)

    #material_number = df.iloc[0]["Material"]

    # Placeholder in order to combine with training data later
    df['actual_duration_nowkend'] = -1

    # Getting train data
    trainfile = "output2.xlsx" #name of file that data will be trained on
    traindata = pd.read_excel(trainfile, sheet_name=group_number)

    # Prepping data to be put into model

    # Adding test datapoint to end of training data, so pd.get_dummies works correctly
    combined = traindata.append(df, sort = False)

    combined = pd.get_dummies(combined, prefix=['MaterialCategory'], columns=['MaterialCategory'])
    combined = pd.get_dummies(combined, prefix=['Group'], columns=['Group'])
    combined = pd.get_dummies(combined, prefix=['Material'], columns=['Material'])
    combined = combined.drop('Confirmation Start Date', 1)
    combined = combined.reset_index(drop=True)

    # Splitting back into train data and test datapoint
    # Datapoint work
    df = combined.iloc[[-1]]
    df = df.drop('actual_duration_nowkend', 1)
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='ignore')  # making sure all values are numeric

    # Traindata work
    traindata = combined.iloc[:-1]
    sub1 = traindata.pop('actual_duration_nowkend')
    traindata['actual_duration_nowkend'] = sub1
    cols = traindata.columns
    traindata[cols] = traindata[cols].apply(pd.to_numeric, errors='ignore')  # making sure all values are numeric

    # Reading in and getting Parameter values
    sheetname = group_number + "_params"
    param_df = pd.read_excel("group_params.xlsx", sheet_name=sheetname)
    param_df = param_df.set_index("Params")

    colsample_bylevel = param_df.loc["colsample_bylevel"]["Values"]
    colsample_bytree = param_df.loc["colsample_bytree"]["Values"]
    gamma = param_df.loc["gamma"]["Values"]
    learning_rate = param_df.loc["learning_rate"]["Values"]
    max_delta_step = param_df.loc["max_delta_step"]["Values"]
    max_depth = int(param_df.loc["max_depth"]["Values"])
    min_child_weight = param_df.loc["min_child_weight"]["Values"]
    subsample = param_df.loc["subsample"]["Values"]

    # Putting parameters in dictionary
    param = {
        "colsample_bylevel": colsample_bylevel,
        "colsample_bytree": colsample_bytree,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_delta_step": max_delta_step,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample
    }

    # Splitting data and labels
    X, y = traindata.iloc[:, :-1], traindata.iloc[:, -1]

    # Calling the prediction interval function which will return prediction and error bound
    alpha = 0.05

    final_result = prediction_interval(param, X, y, df, alpha)

    # Returning the final_result
    return final_result






## Taken from https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/
## Helper method that actually does all predictions/error bounds
def prediction_interval(params, X_train, y_train, datapoint, alpha: float = 0.05):
    #   Compute a prediction interval around the model's prediction of datapoint.

    #   INPUT
    #     params
    #       parameters for XGBoost model
    #     X_train: dataframe of shape (n_samples, n_features)
    #       dataframe containing the training input data
    #     y_train: dataframe of shape (n_samples,)
    #       list of actual durations of train data
    #     datapoint
    #       Datapoint that needs to be predicted of shape (n_features,)
    #     alpha: float = 0.05
    #       The prediction uncertainty. 0.05 means 95% prediction interval

    #   OUTPUT
    #     A triple (`lower`, `pred`, `upper`) with `pred` being the prediction
    #     of the model and `lower` and `upper` constituting the lower- and upper
    #     bounds for the prediction interval around `pred`, respectively. '''

    # Number of training samples
    n = X_train.shape[0]

    # The authors choose the number of bootstrap samples as the square root
    # of the number of samples
    # 2*np.sqrt(n).astype(int)
    nbootstraps = 50

    # Compute the m_i's and the validation residuals (difference between validation pred and actual values)
    bootstrap_preds, val_residuals = np.empty(nbootstraps), []
    for b in range(nbootstraps):
        train_idxs = np.random.choice(range(n), size=n, replace=True).tolist()
        val_idxs = np.array([idx for idx in range(n) if idx not in train_idxs]).tolist()

        dtrain = xgb.DMatrix(X_train.iloc[train_idxs], label=y_train.iloc[train_idxs])
        dtest = xgb.DMatrix(X_train.iloc[val_idxs], label=y_train.iloc[val_idxs])
        # training/model predictions
        num_boost_round = 999
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")], verbose_eval=False,
                          early_stopping_rounds=10)
        num_boost_round = model.best_iteration + 1
        best_model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")],
                               verbose_eval=False)
        best_model.save_model("my_model.model")
        xg_reg = xgb.XGBRegressor()
        xg_reg.load_model("my_model.model")
        preds = xg_reg.predict(X_train.iloc[val_idxs])

        val_residuals.append(y_train.iloc[val_idxs] - preds)
        bootstrap_preds[b] = xg_reg.predict(datapoint)
    bootstrap_preds -= np.mean(bootstrap_preds)
    val_residuals = np.concatenate(val_residuals)

    # Compute the prediction and the training residuals (difference between training pred and actual values)
    # Training/model predictions

    # Train/test split in order to run first train sequence to find optimal num_boost_round

    # ratio = 0.8
    # length = y_train.shape[0]
    # pivot = int(ratio * length)
    #
    # X_newtrain = X_train[:pivot]
    # y_newtrain = y_train[:pivot]
    #
    # X_newtest = X_train[pivot:]
    # y_newtest = y_train[pivot:]
    #
    # dtrain = xgb.DMatrix(X_newtrain, label=y_newtrain)
    # dtest = xgb.DMatrix(X_newtest, label=y_newtest)
    #
    # num_boost_round = 999
    # model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")], verbose_eval=False,
    #                   early_stopping_rounds=10)
    # num_boost_round = model.best_iteration + 1
    # # Now training model on all the data
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # best_model = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
    # best_model.save_model("my_model.model")
    # xg_reg = xgb.XGBRegressor()
    # xg_reg.load_model("my_model.model")
    #
    # # Getting training residuals
    # preds = xg_reg.predict(X_train)
    # train_residuals = y_train - preds
    #
    # # Take percentiles of the training- and validation residuals to enable
    # # comparisons between them
    # val_residuals = np.percentile(val_residuals, q=np.arange(100))
    # train_residuals = np.percentile(train_residuals, q=np.arange(100))
    #
    # # Compute the .632+ bootstrap estimate for the sample noise and bias (effectively combining training/validation residuals
    # # in best possible ratio to prevent over/under estimating)
    # no_information_error = np.mean(np.abs(np.random.permutation(y_train) - \
    #                                       np.random.permutation(preds)))
    # generalisation = np.abs(val_residuals - train_residuals)
    # no_information_val = np.abs(no_information_error - train_residuals)
    # relative_overfitting_rate = np.mean(generalisation / no_information_val)
    # weight = .632 / (1 - .368 * relative_overfitting_rate)
    # residuals = (1 - weight) * train_residuals + weight * val_residuals

    # Construct the C set and get the percentiles
    # (Combining residuals and model variance to create final distribution(C set) of possible prediction values,
    #  then take percentiles depending on alpha for lower/upper bounds)
    C = np.array([m + o for m in bootstrap_preds for o in val_residuals])
    #C = np.sort(C)
    qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    percentiles = np.percentile(C, q=qs)
    #percentiles2 = np.percentile(C, q=[0, 100])
    pred = xg_reg.predict(datapoint)

    #     plt.hist(bootstrap_preds)
    #     plt.hist(residuals)
    #     plt.hist(C)
    #     plt.show()
    # display(percentiles2)

    bottom = pred - abs(percentiles[0])
    upper = pred + abs(percentiles[1])

    if bottom <= 0:
        bottom = 1.0

    return pd.Series([pred, bottom, upper])

if __name__ == '__main__':
    app.run()