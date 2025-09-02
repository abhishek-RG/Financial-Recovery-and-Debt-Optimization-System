from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os

# Set default plotly template to dark theme
pio.templates.default = "plotly_dark"

app = Flask(__name__)
app.secret_key = os.urandom(24) # More secure secret key

# ----------------------
# Utilities
# ----------------------

def load_csv(file):
    """Load csv from Werkzeug FileStorage."""
    try:
        return pd.read_csv(file)
    except Exception:
        file.stream.seek(0)
        return pd.read_csv(file.stream)

def detect_date_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "month", "period", "time", "timestamp"]):
            return c
    return None

def create_time_features(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df = df.reset_index(drop=True)
    return df

def safe_first_match(cols, keywords):
    for k in keywords:
        for c in cols:
            if k in c.lower():
                return c
    return None

def get_model_and_scaler(df, cash_col, date_col, model_choice, test_size):
    working = df.copy()

    # Feature engineering: lags + rolling mean
    lags = 3
    for l in range(1, lags + 1):
        working[f'lag_{l}'] = working[cash_col].shift(l)
    working['rolling_3'] = working[cash_col].rolling(window=3, min_periods=1).mean()
    if date_col:
        working = create_time_features(working, date_col)

    working = working.dropna().reset_index(drop=True)

    feature_cols = [c for c in working.columns
                    if c.startswith('lag_') or c.startswith('rolling_') or c in ['month','quarter']]
    X = working[feature_cols].select_dtypes(include=[np.number])
    y = working[cash_col].values

    if X.shape[0] < 10:
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size)/100.0, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_choice == 'LinearRegression':
        model = LinearRegression()
    elif model_choice == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=200, random_state=42)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    r2 = r2_score(y_test, preds)

    return model, scaler, feature_cols, r2, working

# ----------------------
# Routes (views)
# ----------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/cashflow", methods=["GET", "POST"])
def cashflow():
    if request.method == "GET":
        return render_template("cashflow.html", model_choice="RandomForest", test_size=20, periods=6)

    file = request.files.get('file')
    model_choice = request.form.get('model_choice', 'RandomForest')
    test_size = int(request.form.get('test_size', 20))
    periods = int(request.form.get('periods', 6))

    if not file or file.filename == '':
        flash("Please upload a CSV file to begin analysis.", "warning")
        return redirect(url_for("cashflow"))

    df = load_csv(file)
    if df.empty:
        flash("Uploaded file is empty or invalid.", "danger")
        return redirect(url_for("cashflow"))

    date_col = detect_date_column(df)
    cash_col = safe_first_match(df.columns, ["net_cash","cashflow","netcash","net cash"])
    inflow_col = safe_first_match(df.columns, ["inflow","cash_in","receipt","amount_received"])
    outflow_col = safe_first_match(df.columns, ["outflow","cash_out","payment","amount_paid"])

    if cash_col is None:
        if inflow_col and outflow_col:
            df['net_cash'] = pd.to_numeric(df[inflow_col], errors='coerce').fillna(0) - \
                             pd.to_numeric(df[outflow_col], errors='coerce').fillna(0)
            cash_col = 'net_cash'
        else:
            flash("Could not find 'net_cash' or 'inflow'/'outflow' columns.", "danger")
            return redirect(url_for("cashflow"))

    if date_col:
        df = create_time_features(df, date_col)

    df[cash_col] = pd.to_numeric(df[cash_col], errors='coerce')
    df = df.dropna(subset=[cash_col]).reset_index(drop=True)

    # KPIs
    avg_cash = float(df[cash_col].mean())
    recent_mean = float(df[cash_col].tail(6).mean() if len(df) >= 6 else df[cash_col].mean())
    neg_months = int((df[cash_col] < 0).sum())
    pct_negative = neg_months / max(1, len(df))
    last_value = float(df[cash_col].iloc[-1])

    kpis = dict(avg_cash=avg_cash, recent_mean=recent_mean, neg_months=neg_months,
                total_rows=len(df), last_value=last_value)

    alerts = []
    if avg_cash < 0:
        alerts.append({"cls":"danger","msg":"Average net cashflow is NEGATIVE — urgent action required."})
    elif pct_negative > 0.4:
        alerts.append({"cls":"warning","msg":"High percentage of negative cashflow periods detected."})
    if last_value < 0:
        alerts.append({"cls":"warning","msg":"Latest cashflow is negative — cash burn detected."})

    # Visuals
    fig_line = px.line(df, x=date_col if date_col else df.index, y=cash_col, title="Cashflow Over Time")
    fig_line.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    graph_cash = pio.to_html(fig_line, full_html=False, include_plotlyjs='cdn')

    cash_flow_counts = (df[cash_col] > 0).value_counts()
    pie_data = pd.DataFrame({'status': ['Positive' if k else 'Negative' for k in cash_flow_counts.index],
                             'count': cash_flow_counts.values})
    fig_pie = px.pie(pie_data, names='status', values='count', title='Positive vs Negative Periods', hole=0.4)
    fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    graph_pie = pio.to_html(fig_pie, full_html=False, include_plotlyjs='cdn')

    graph_bar = None
    if date_col and 'month' in df.columns:
        monthly_cash_avg = df.groupby('month')[cash_col].mean().reset_index()
        fig_bar = px.bar(monthly_cash_avg, x='month', y=cash_col, title='Average Monthly Cash Flow (Seasonality)', text_auto='.2s')
        fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        graph_bar = pio.to_html(fig_bar, full_html=False, include_plotlyjs='cdn')

    # Model + forecast
    model, scaler, feature_cols, r2, working_df = get_model_and_scaler(df, cash_col, date_col, model_choice, test_size)
    graph_forecast, runway = None, None

    if model and scaler and feature_cols and working_df is not None:
        last_row = working_df.iloc[-1:].copy()
        cur_feats = last_row[feature_cols].copy()
        future = []
        for _ in range(int(periods)):
            cur_s = scaler.transform(cur_feats)
            fval = float(model.predict(cur_s)[0])
            future.append(fval)

            lag_cols = sorted([c for c in feature_cols if c.startswith('lag_')])
            for idx in range(len(lag_cols)-1, 0, -1):
                cur_feats[lag_cols[idx]] = cur_feats[lag_cols[idx-1]]
            if lag_cols:
                cur_feats[lag_cols[0]] = fval
            if 'rolling_3' in cur_feats.columns:
                cur_feats['rolling_3'] = (cur_feats.get('rolling_3', 0) * 2 + fval) / 3

        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            freq = pd.infer_freq(df[date_col].dropna())
            freq = freq if freq else 'M'
            forecast_dates = pd.date_range(df[date_col].dropna().iloc[-1], periods=periods+1, freq=freq)[1:]
            forecast_df = pd.DataFrame({date_col: forecast_dates, cash_col: future})
        else:
            forecast_df = pd.DataFrame({cash_col: future})

        hist = df[[date_col, cash_col]] if date_col in df.columns else df[[cash_col]]
        hist = hist.assign(type='Historical')
        fc = forecast_df.assign(type='Forecast')
        combined = pd.concat([hist, fc], ignore_index=True)

        fig_for = px.line(combined, x=date_col if date_col in combined.columns else combined.index,
                          y=cash_col, color='type',
                          title=f"Cash Flow: Historical & {periods}-Period Forecast")
        fig_for.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        graph_forecast = pio.to_html(fig_for, full_html=False, include_plotlyjs='cdn')

    table_html = df.head(10).to_html(classes="table table-hover", index=False)

    return render_template("cashflow.html",
                           model_choice=model_choice, test_size=test_size, periods=periods,
                           kpis=kpis, alerts=alerts, graph_cash=graph_cash,
                           graph_pie=graph_pie, graph_bar=graph_bar, r2=r2,
                           graph_forecast=graph_forecast, table_html=table_html,
                           file_uploaded=True)

@app.route("/debt", methods=["GET", "POST"])
def debt():
    if request.method == "GET":
        return render_template("debt.html")

    file = request.files.get('file')
    if not file or file.filename == '':
        flash("Please upload a CSV file to begin analysis.", "warning")
        return redirect(url_for("debt"))

    df = load_csv(file)
    if df.empty:
        flash("Uploaded file is empty or invalid.", "danger")
        return redirect(url_for("debt"))

    principal_col = safe_first_match(df.columns, ["principal","loan_amount","amount"])
    outstanding_col = safe_first_match(df.columns, ["outstanding","balance","remaining"])
    rate_col = safe_first_match(df.columns, ["interest","rate","interest_rate","interest_rate(%)"])
    emi_col = safe_first_match(df.columns, ["emi","monthly_installment","monthly"])

    if principal_col is None or rate_col is None:
        flash("Could not detect 'Principal' and 'Interest Rate' columns.", "danger")
        return redirect(url_for("debt"))

    df[principal_col] = pd.to_numeric(df[principal_col], errors='coerce').fillna(0)
    df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce').fillna(0)

    if outstanding_col:
        df[outstanding_col] = pd.to_numeric(df[outstanding_col], errors='coerce').fillna(df[principal_col])
    else:
        df['Outstanding_Amount'] = df[principal_col]
        outstanding_col = 'Outstanding_Amount'

    if emi_col:
        df[emi_col] = pd.to_numeric(df[emi_col], errors='coerce').fillna((df[principal_col]/12).astype(int))
    else:
        df['Monthly_Installment'] = (df[principal_col]/12).astype(int)
        emi_col = 'Monthly_Installment'

    # KPIs
    total_principal = float(df[principal_col].sum())
    total_outstanding = float(df[outstanding_col].sum())
    total_emi = float(df[emi_col].sum())
    debt_kpis = dict(total_principal=total_principal, total_outstanding=total_outstanding, total_emi=total_emi)

    # Alerts
    high_rate_count = int((df[rate_col] > 20).sum())
    alerts = []
    if high_rate_count > 0:
        alerts.append({"cls":"warning", "msg":f"{high_rate_count} loan(s) have interest > 20% — consider refinancing."})

    # Chart: rate distribution
    fig_rate = px.histogram(df, x=rate_col, nbins=30, title="Interest Rate Distribution")
    fig_rate.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    graph_rate_dist = pio.to_html(fig_rate, full_html=False, include_plotlyjs='cdn')

    table_html = df.head(10).to_html(classes="table table-hover", index=False)

    return render_template("debt.html",
                           debt_kpis=debt_kpis, alerts=alerts,
                           graph_rate_dist=graph_rate_dist,
                           table_html=table_html,
                           file_uploaded=True)


# JSON API (remains unchanged)
@app.route("/api/cashflow/analyze", methods=["POST"])
def api_cashflow_analyze():
    file = request.files.get('file')
    if not file:
        return jsonify(error="file missing"), 400

    model_choice = request.form.get('model_choice', 'RandomForest')
    test_size = int(request.form.get('test_size', 20))
    periods = int(request.form.get('periods', 6))

    df = load_csv(file)
    date_col = detect_date_column(df)
    cash_col = safe_first_match(df.columns, ["net_cash","cashflow","netcash","net cash"])
    inflow_col = safe_first_match(df.columns, ["inflow","cash_in","receipt","amount_received"])
    outflow_col = safe_first_match(df.columns, ["outflow","cash_out","payment","amount_paid"])
    if cash_col is None:
        if inflow_col and outflow_col:
            df['net_cash'] = pd.to_numeric(df[inflow_col], errors='coerce').fillna(0) - \
                             pd.to_numeric(df[outflow_col], errors='coerce').fillna(0)
            cash_col = 'net_cash'
        else:
            return jsonify(error="cash column not found"), 400
    if date_col:
        df = create_time_features(df, date_col)
    df[cash_col] = pd.to_numeric(df[cash_col], errors='coerce')
    df = df.dropna(subset=[cash_col]).reset_index(drop=True)

    avg_cash = float(df[cash_col].mean())
    recent_mean = float(df[cash_col].tail(6).mean() if len(df) >= 6 else df[cash_col].mean())
    neg_months = int((df[cash_col] < 0).sum())
    last_value = float(df[cash_col].iloc[-1])
    kpis = dict(avg_cash=avg_cash, recent_mean=recent_mean, neg_months=neg_months,
                total_rows=len(df), last_value=last_value)

    model, scaler, feature_cols, r2, working_df = get_model_and_scaler(df, cash_col, date_col, model_choice, test_size)

    forecast = []
    if model and scaler and feature_cols and working_df is not None:
        last_row = working_df.iloc[-1:].copy()
        cur_feats = last_row[feature_cols].copy()
        for _ in range(int(periods)):
            cur_s = scaler.transform(cur_feats)
            fval = float(model.predict(cur_s)[0])
            forecast.append(fval)
            lag_cols = sorted([c for c in feature_cols if c.startswith('lag_')])
            for idx in range(len(lag_cols)-1, 0, -1):
                cur_feats[lag_cols[idx]] = cur_feats[lag_cols[idx-1]]
            if lag_cols:
                cur_feats[lag_cols[0]] = fval
            if 'rolling_3' in cur_feats.columns:
                cur_feats['rolling_3'] = (cur_feats.get('rolling_3', 0) * 2 + fval) / 3

    return jsonify(
        kpis=kpis,
        r2=r2 if r2 is not None else None,
        series=df[cash_col].tolist(),
        forecast=forecast
    )

if __name__ == "__main__":
    app.run(debug=True)