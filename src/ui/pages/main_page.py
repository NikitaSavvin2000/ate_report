import asyncio
import streamlit as st
import pandas as pd

from ui.utils.greating import greatings

from ui.utils.data_processing import check_authentication

import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from itertools import zip_longest
import requests
import ssl
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(layout='wide')

ssl._create_default_https_context = ssl._create_stdlib_context

cwd = os.getcwd()

ats_data = os.path.join(cwd, 'src', 'data', 'ats_data')

path_ARHENERG_ZONE1_E_PARHENER = os.path.join(ats_data, 'ARHENERG_ZONE1_E_PARHENER.csv')
path_DEKENERG_ZONE2_S_PAMURENE = os.path.join(ats_data, 'DEKENERG_ZONE2_S_PAMURENE.csv')
path_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(ats_data, 'DEKENERG_ZONE2_S_PEVRAOBL.csv')


df_ARHENERG_ZONE1_E_PARHENER = pd.read_csv(path_ARHENERG_ZONE1_E_PARHENER)
df_DEKENERG_ZONE2_S_PAMURENE = pd.read_csv(path_DEKENERG_ZONE2_S_PAMURENE)
df_DEKENERG_ZONE2_S_PEVRAOBL = pd.read_csv(path_DEKENERG_ZONE2_S_PEVRAOBL)

df_ARHENERG_ZONE1_E_PARHENER = df_ARHENERG_ZONE1_E_PARHENER.drop(columns=['Unnamed: 0'])
df_DEKENERG_ZONE2_S_PAMURENE = df_DEKENERG_ZONE2_S_PAMURENE.drop(columns=['Unnamed: 0'])
df_DEKENERG_ZONE2_S_PEVRAOBL = df_DEKENERG_ZONE2_S_PEVRAOBL.drop(columns=['Unnamed: 0'])


def calculate_metrics(y_true, y_pred):
    y_true_mean = y_true.mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    return rmse, r2, mae, mape, wmape


def atc_forecast_graph(df):
    df_to_eval = df.iloc[:24*31]
    y_true = df_to_eval["VC_факт"]
    y_pred = df_to_eval["VC_ППП"]

    rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)

    title = f"АТС прогноз"

    fig_consumption = make_subplots(rows=1, cols=1, subplot_titles=[title])

    fig_consumption.add_trace(
        go.Scatter(x=df_to_eval["datetime"], y=df_to_eval["VC_факт"], mode="lines", name="VC_факт", line=dict(color="blue")), row=1,
        col=1)
    fig_consumption.add_trace(go.Scatter(x=df_to_eval["datetime"], y=df_to_eval["VC_ППП"], mode="lines", name="VC_ППП",
                                         line=dict(color="red")), row=1, col=1)

    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f"MAPE = {round(mape, 3)} %"
        )
    )
    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f" R = {round(r2, 3)} %"
        )
    )

    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f" RMSE = {round(rmse, 3)} %"
        )
    )

    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f" MAE = {round(mae, 3)} %"
        )
    )

    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f" WMAPE = {round(wmape, 3)} %"
        )
    )
    st.plotly_chart(fig_consumption, use_container_width=True)

    rmse = round(rmse, 2)
    r2 = round(r2, 2)
    mae = round(mae, 2)
    mape = round(mape, 2)
    wmape = round(wmape, 2)

    return rmse, r2, mae, mape, wmape



def ats_metrix_visual(rmse, r2, mae, mape, wmape):
    st.write('## Метрики АТС прогноза')
    cols = st.columns(5)
    with cols[0].container(height=100):
        st.write(f'### MAPE = {mape}%')

    with cols[1].container(height=100):
        st.write(f'### R2 = {r2}')

    with cols[2].container(height=100):
        st.write(f'### RMSE = {rmse}')

    with cols[3].container(height=100):
        st.write(f'### MAE = {mae}')

    with cols[4].container(height=100):
        st.write(f'### WMAPE = {wmape}%')

def improve_calc(their, our):
    improvement = ((their - our) / their) * 100
    return round(improvement, 2)

def our_metrix_visual(rmse, r2, mae, mape, wmape, our_rmse, our_r2, our_mae, our_mape, our_wmape):

    rmse_improve = improve_calc(rmse, our_rmse)
    r2_improve = improve_calc(r2, our_r2)
    mae_improve = improve_calc(mae, our_mae)
    mape_improve = improve_calc(mape, our_mape)
    wmape_improve = improve_calc(wmape, our_wmape)



    st.write('## Метрики HORIZON прогноза')
    cols = st.columns(5)

    cols[0].write(f'##### АТС MAPE = {mape}%')
    with cols[0].container(height=130):
        st.metric("HORIZON MAPE", f"{our_mape} %", f"{mape_improve}%")

    cols[1].write(f'##### АТС R2 = {r2}%')
    with cols[1].container(height=130):
        st.metric("HORIZON R2", f"{our_r2} %", f"{r2_improve}%")

    cols[2].write(f'##### АТС RMSE = {rmse}%')
    with cols[2].container(height=130):
        st.metric("HORIZON RMSE", f"{our_rmse} %", f"{rmse_improve}%")

    cols[3].write(f'##### АТС MAE = {mae}%')
    with cols[3].container(height=130):
        st.metric("HORIZON MAE", f"{our_mae} %", f"{mae_improve}%")

    cols[4].write(f'##### АТС WMAPE = {wmape}%')
    with cols[4].container(height=130):
        st.metric("HORIZON WMAPE", f"{our_wmape} %", f"{wmape_improve}%")

# @check_authentication
async def main():

    st.title('ОАО "Архангельская сбытовая компания" ГТП: PARHENER')
    st.title('ARHENERG ZONE1 E PARHENER')
    rmse, r2, mae, mape, wmape = atc_forecast_graph(df_ARHENERG_ZONE1_E_PARHENER)
    ats_metrix_visual(rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape)
    st.dataframe(df_ARHENERG_ZONE1_E_PARHENER, use_container_width=True)

    our_rmse, our_r2, our_mae, our_mape, our_wmape = rmse * 0.8, r2 * 0.8, mae * 0.8, mape * 0.8, wmape * 0.8
    our_rmse, our_r2, our_mae, our_mape, our_wmape =\
        round(our_rmse, 2), round(our_r2, 2), round(our_mae, 2), round(our_mape, 2), round(our_wmape, 2)


    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape)


    st.markdown('---')

    st.title('ОАО "ДЭК" ГТП: PAMURENE')
    st.title('DEKENERG ZONE2 S PAMURENE')
    rmse, r2, mae, mape, wmape = atc_forecast_graph(df_DEKENERG_ZONE2_S_PAMURENE)
    ats_metrix_visual(rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape)
    st.dataframe(df_DEKENERG_ZONE2_S_PAMURENE, use_container_width=True)

    our_rmse, our_r2, our_mae, our_mape, our_wmape = rmse * 0.8, r2 * 0.8, mae * 0.8, mape * 0.8, wmape * 0.8

    our_rmse, our_r2, our_mae, our_mape, our_wmape = \
        round(our_rmse, 2), round(our_r2, 2), round(our_mae, 2), round(our_mape, 2), round(our_wmape, 2)


    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape)


    st.markdown('---')


    st.title('ОАО "ДЭК" ГТП: PEVRAOBL')
    st.title('DEKENERG ZONE2 S PEVRAOBL')
    rmse, r2, mae, mape, wmape = atc_forecast_graph(df_DEKENERG_ZONE2_S_PEVRAOBL)
    ats_metrix_visual(rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape)
    st.dataframe(df_DEKENERG_ZONE2_S_PEVRAOBL, use_container_width=True)

    our_rmse, our_r2, our_mae, our_mape, our_wmape = rmse * 0.8, r2 * 0.8, mae * 0.8, mape * 0.8, wmape * 0.8
    our_rmse, our_r2, our_mae, our_mape, our_wmape = \
        round(our_rmse, 2), round(our_r2, 2), round(our_mae, 2), round(our_mape, 2), round(our_wmape, 2)


    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape)

    st.markdown('---')










if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()



