import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import datetime
from pandas.plotting import register_matplotlib_converters

cons = (
    {'type': 'ineq', 'fun': lambda x: np.min(x)},
    {'type': 'eq',  'fun': lambda x: np.sum(x)-1}
)

def portfolio_var(weight, returns):
    cov = returns.cov()
    return np.dot(np.dot(weight, cov), weight.T)*10000

def minimize_var(returns, weight0):
    if returns.shape[0]<2:
        return 0, np.repeat(1/returns.shape[1], returns.shape[1])
    else:
        # SciPyを使って最適化を行う
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        res = minimize(portfolio_var, weight0, constraints=cons, method="SLSQP", args=(returns))
        return res.fun/10000, res.x

if __name__ == "__main__":
    register_matplotlib_converters()

    # Inputs
    # 調べたいものを入れます
    symbols    = ["SPXL", "TMF"]
    date_start = "2010-04-01"
    date_end   = datetime.datetime.today() # "2020-04-01" などの日時、デフォルトはtoday

    # Get Price and Return
    # Yahoo Financeから指定したシンボルをダウンロード
    prices = yf.download(symbols, date_start)
    # シンボルで指定した内容を全て小文字から大文字に変換する
    # 大文字に変換するのはデータの指定の際に大文字小文字の違いによってデータ取得に失敗する可能性があるからだと推測される
    symbols = [x.upper() for x in symbols]
    # ダウンロードしたデータの終値を"Adj Close"で指定する
    # pct_changesやilocはpandasの機能で、データの整形を行っている
    # https://note.nkmk.me/python-pandas-diff-pct-change/
    # https://note.nkmk.me/python-pandas-at-iat-loc-iloc/
    returns = prices["Adj Close"][symbols].pct_change().iloc[1:]

    # Next Month Weight (if day > 21)
    # 取得したデータの最後の日付が21以上なら後に出てくるifで処理を追加する
    # nextmonthdayは20日後の日付を格納し、後のifの処理中で使用する
    LastDay = returns.index[-1]
    NextMonth = LastDay.day > 21
    nextmonthday = LastDay + datetime.timedelta(days=20)
    n_y, n_m = nextmonthday.year, nextmonthday.month

    # Re-index
    returns = returns.set_index([returns.index.year, returns.index.month, returns.index])
    returns.index.names=["year", "month", "date"]

    # Optimization
    # weight0というarrayを用意する
    # https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
    weight0 = np.repeat(1/len(symbols),len(symbols))
    # pandasで年月でまとめたうえで(groupby)、最適化を行う(apply)
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.GroupBy.apply.html
    res = returns.groupby(["year", "month"]).apply(minimize_var, weight0=weight0)

    # Return
    # indexを作成する
    ind = ["{0}-{1}".format(x[0], x[1]) for x in res.index]
    # 値が入っていない場合には0で埋める
    var_df = pd.DataFrame({"var": [x[0] for x in res]}, index =ind).fillna(0)
    # resの一部の値をnumpyのarray型に変換する
    w = np.array([x[1] for x in res])
    # indexの日付を文字列に変換する
    w_ind = [str(x) for x in pd.to_datetime(ind)]
    # pandasのdataframeに変換する
    w_df = pd.DataFrame(w, index=w_ind, columns=symbols)
    # 21日以降であった場合、データを追加する
    if NextMonth:
        s = pd.Series(np.repeat(np.nan, len(w_df.columns)), index=w_df.columns, name=str(pd.to_datetime("{0}-{1}".format(n_y, n_m))))
        w_df = w_df.append(s)
    # 1行下方向にずらして、NaNの値を埋める
    # https://note.nkmk.me/python-pandas-shift/
    w_df = w_df.shift(1).fillna(1/len(symbols))

    # Portfolio total return
    # pandasを使用して月単位の積を求めることで、月次リターンを作成する
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.GroupBy.prod.html
    monthly_returns_df = (returns+1).groupby(["year", "month"]).prod()
    # ポートフォリオのリターンは月次リターンと重み付けをかけることで求める
    portfolio_return_df = (monthly_returns_df - 1) * w_df.values[0:len(monthly_returns_df)]
    # リターンのdataframeにインデックスを追加する
    return_df = pd.DataFrame({"return":portfolio_return_df.sum(axis=1).values},
                             index=ind)
    # 累積和を求める
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cumprod.html
    cum_return = (1+return_df).cumprod()

    # グラフ作成
    plt.rcParams["figure.facecolor"] = 'w'
    fig, axes = plt.subplots(4,1, gridspec_kw=dict(height_ratios=[2,2,1,1], hspace=0.3), figsize=(12,10))

    axes[0].stackplot(pd.to_datetime(w_df.index), w_df.values.T*100, labels=symbols, alpha=0.8)
    handles, labels = axes[0].get_legend_handles_labels()

    axes[0].legend(handles[::-1], labels[::-1], loc = 'upper left', bbox_to_anchor = (1.05, 1), title="Symbols")
    axes[0].set_title("Minimum Variance Weights")

    axes[1].plot(cum_return)
    axes[1].set_yscale("log")
    axes[1].set_title("Cumulative Return")

    axes[2].plot(np.sqrt(var_df*252)*100)
    axes[2].set_title("Analized STDDEV (%)")

    axes[3].plot(return_df*100)
    axes[3].set_title("Monthly Return (%)")

    # Jupyterではなくローカルで動かした際にグラフを表示する
    plt.show()

    # Output JSON
    # グラフ化するのではなくJSONで出力する
    w_df.index = w_df.index.astype("str")
    js = {"date": [str(x) for x in ind],
          "a_std":[str(np.round(np.sqrt(var_df.values[i][0]*252)*100,2)) for i in range(len(var_df))],
          "m_ret":[str(np.round(return_df.values[i][0]*100,2)) for i in range(len(return_df))],
          "cum_ret":[str(np.round(cum_return["return"][i], 4)) for i in range(len(cum_return))],
         "mvweight":np.round(w_df,4).astype("str").T.to_dict(orient="split")}
    print(json.dumps(js, indent=4))
