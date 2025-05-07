import asyncio
import streamlit as st

import ssl
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(layout='wide')

ssl._create_default_https_context = ssl._create_stdlib_context


ats_data = os.path.join('src', 'data', 'ats_data')

# path_ARHENERG_ZONE1_E_PARHENER = os.path.join(ats_data, 'ARHENERG_ZONE1_E_PARHENER.csv')
path_DEKENERG_ZONE2_S_PAMURENE = os.path.join(ats_data, 'DEKENERG_ZONE2_S_PAMURENE.csv')
path_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(ats_data, 'DEKENERG_ZONE2_S_PEVRAOBL.csv')


# path_to_read_cast_1 = "/Users/nikitasavvin/Desktop/Business/repo/ate_report/src/data/AutoNHITS/DEKENERG_ZONE2_S_PAMURENE_AutoNHITS_predict.xlsx"
# path_to_read_cast_2 = "/Users/nikitasavvin/Desktop/Business/repo/ate_report/src/data/AutoNHITS/DEKENERG_ZONE2_S_PEVRAOBL_AutoNHITS_predict.xlsx"
#
# df1 = pd.read_excel(path_to_read_cast_1)
# df2 = pd.read_excel(path_to_read_cast_2)
# st.write(df1)


# df1 = df1.rename(columns={"VC_факт_pred": "consumption", "ds": "Datetime",})
# df1 = df1.drop(columns=["unique_id", "VC_факт", "VC_ППП", "minute", "dayofweek","dayofmonth","day_zone_encoded",])
# st.write(df1.columns)
# df2 = df2.rename(columns={"VC_факт_pred": "consumption", "ds": "Datetime",})
# df2 = df2.drop(columns=["unique_id", "VC_факт", "I_откл ph", "second", "VC_ППП"])

# df1.to_csv(path1)
# df2.to_csv(path2)

# path_to_save_cast_1 = "/Users/nikitasavvin/Desktop/Business/repo/ate_report/src/data/AutoNHITS/DEKENERG_ZONE2_S_PAMURENE_AutoNHITS_predict.csv"
# path_to_save_cast_2 = "/Users/nikitasavvin/Desktop/Business/repo/ate_report/src/data/AutoNHITS/DEKENERG_ZONE2_S_PEVRAOBL_AutoNHITS_predict.csv"
#
# df1.to_csv(path_to_save_cast_1)
# df2.to_csv(path_to_save_cast_2)


# df_ARHENERG_ZONE1_E_PARHENER = pd.read_csv(path_ARHENERG_ZONE1_E_PARHENER)
df_DEKENERG_ZONE2_S_PAMURENE = pd.read_csv(path_DEKENERG_ZONE2_S_PAMURENE)
df_DEKENERG_ZONE2_S_PEVRAOBL = pd.read_csv(path_DEKENERG_ZONE2_S_PEVRAOBL)

# df_ARHENERG_ZONE1_E_PARHENER = df_ARHENERG_ZONE1_E_PARHENER.drop(columns=['Unnamed: 0'])
df_DEKENERG_ZONE2_S_PAMURENE = df_DEKENERG_ZONE2_S_PAMURENE.drop(columns=['Unnamed: 0'])
df_DEKENERG_ZONE2_S_PEVRAOBL = df_DEKENERG_ZONE2_S_PEVRAOBL.drop(columns=['Unnamed: 0'])

LSTM_path = os.path.join('src', 'data', 'LSTM')
Bi_LSTM_path = os.path.join('src', 'data', 'Bi_LSTM')
CNN_Bi_LSTM_path = os.path.join('src', 'data', 'CNN_Bi_LSTM')
CNN_LSTM_path = os.path.join('src', 'data', 'CNN_LSTM')
XGBoost_path = os.path.join('src', 'data', 'XGBoost')
AutoNHITS_path = os.path.join('src', 'data', 'AutoNHITS')

# path_LSTM_ARHENERG_ZONE1_E_PARHENER = os.path.join(LSTM_path, 'ARHENERG_ZONE1_E_PARHENER_LSTM_predict.csv')
path_LSTM_DEKENERG_ZONE2_S_PAMURENE = os.path.join(LSTM_path, 'DEKENERG_ZONE2_S_PAMURENE_LSTM_predict.csv')
path_LSTM_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(LSTM_path, 'DEKENERG_ZONE2_S_PEVRAOBL_LSTM_predict.csv')

# path_BI_LSTM_ARHENERG_ZONE1_E_PARHENER = os.path.join(Bi_LSTM_path, 'ARHENERG_ZONE1_E_PARHENER_Bi_LSTM_predict.csv')
path_BI_LSTM_DEKENERG_ZONE2_S_PAMURENE = os.path.join(Bi_LSTM_path, 'DEKENERG_ZONE2_S_PAMURENE_Bi_LSTM_predict.csv')
path_BI_LSTM_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(Bi_LSTM_path, 'DEKENERG_ZONE2_S_PEVRAOBL_Bi_LSTM_predict.csv')

# path_CNN_Bi_LSTM_ARHENERG_ZONE1_E_PARHENER = os.path.join(CNN_Bi_LSTM_path, 'ARHENERG_ZONE1_E_PARHENER_CNN-Bi-LSTM_predict.csv')
path_CNN_Bi_LSTM_DEKENERG_ZONE2_S_PAMURENE = os.path.join(CNN_Bi_LSTM_path, 'DEKENERG_ZONE2_S_PAMURENE_CNN_BI_LSTM_predict.csv')
path_CNN_Bi_LSTM_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(CNN_Bi_LSTM_path, 'DEKENERG_ZONE2_S_PEVRAOBL_CNN_BI_LSTM_predict.csv')

# path_CNN_LSTM_ARHENERG_ZONE1_E_PARHENER = os.path.join(CNN_LSTM_path, 'ARHENERG_ZONE1_E_PARHENER_CNN-LSTM_predict.csv')
path_CNN_LSTM_DEKENERG_ZONE2_S_PAMURENE = os.path.join(CNN_LSTM_path, 'DEKENERG_ZONE2_S_PAMURENE_CNN_LSTM_predict.csv')
path_CNN_LSTM_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(CNN_LSTM_path, 'DEKENERG_ZONE2_S_PEVRAOBL_CNN_LSTM_predict.csv')

path_XGBoost_DEKENERG_ZONE2_S_PAMURENE = os.path.join(XGBoost_path, 'DEKENERG_ZONE2_S_PAMURENE_XGBoost_predict.scv')
path_XGBoost_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(XGBoost_path, "DEKENERG_ZONE2_S_PEVRAOBL_XGBoost_predict.scv")

path_AutoNHITS_DEKENERG_ZONE2_S_PAMURENE = os.path.join(AutoNHITS_path, 'DEKENERG_ZONE2_S_PAMURENE_AutoNHITS_predict.csv')
path_AutoNHITS_DEKENERG_ZONE2_S_PEVRAOBL = os.path.join(AutoNHITS_path, "DEKENERG_ZONE2_S_PEVRAOBL_AutoNHITS_predict.csv")


# predict_LSTM_df_ARHENERG_ZONE1_E_PARHENER_df = pd.read_csv(path_LSTM_ARHENERG_ZONE1_E_PARHENER)
predict_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df = pd.read_csv(path_LSTM_DEKENERG_ZONE2_S_PAMURENE)
predict_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df = pd.read_csv(path_LSTM_DEKENERG_ZONE2_S_PEVRAOBL)

# predict_BI_LSTM_df_ARHENERG_ZONE1_E_PARHENER_df = pd.read_csv(path_BI_LSTM_ARHENERG_ZONE1_E_PARHENER)
predict_BI_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df = pd.read_csv(path_BI_LSTM_DEKENERG_ZONE2_S_PAMURENE)
predict_BI_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df  = pd.read_csv(path_BI_LSTM_DEKENERG_ZONE2_S_PEVRAOBL)

# predict_CNN_Bi_LSTM_df_ARHENERG_ZONE1_E_PARHENER_df = pd.read_csv(path_CNN_Bi_LSTM_ARHENERG_ZONE1_E_PARHENER)
predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df = pd.read_csv(path_CNN_Bi_LSTM_DEKENERG_ZONE2_S_PAMURENE)
predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df  = pd.read_csv(path_CNN_Bi_LSTM_DEKENERG_ZONE2_S_PEVRAOBL)

# predict_CNN_LSTM_df_ARHENERG_ZONE1_E_PARHENER_df = pd.read_csv(path_CNN_LSTM_ARHENERG_ZONE1_E_PARHENER)
predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df = pd.read_csv(path_CNN_LSTM_DEKENERG_ZONE2_S_PAMURENE)
predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df  = pd.read_csv(path_CNN_LSTM_DEKENERG_ZONE2_S_PEVRAOBL)

predict_XGBoost_df_DEKENERG_ZONE2_S_PAMURENE_df = pd.read_csv(path_XGBoost_DEKENERG_ZONE2_S_PAMURENE)
predict_XGBoost_df_DEKENERG_ZONE2_S_PEVRAOBL_df = pd.read_csv(path_XGBoost_DEKENERG_ZONE2_S_PEVRAOBL)

predict_AutoNHITS_df_DEKENERG_ZONE2_S_PAMURENE_df = pd.read_csv(path_AutoNHITS_DEKENERG_ZONE2_S_PAMURENE)
predict_AutoNHITS_df_DEKENERG_ZONE2_S_PEVRAOBL_df = pd.read_csv(path_AutoNHITS_DEKENERG_ZONE2_S_PEVRAOBL)


df_all_DEKENERG_ZONE2_S_PAMURENE = df_DEKENERG_ZONE2_S_PAMURENE.copy()

df_all_DEKENERG_ZONE2_S_PAMURENE['datetime'] = pd.to_datetime(df_all_DEKENERG_ZONE2_S_PAMURENE['datetime'], format='%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PAMURENE['datetime_str'] = df_all_DEKENERG_ZONE2_S_PAMURENE['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

predict_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'] = pd.to_datetime(predict_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['datetime_str'] = predict_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.merge(predict_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.rename(columns={"consumption": "LSTM"})

predict_BI_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'] = pd.to_datetime(predict_BI_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_BI_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['datetime_str'] = predict_BI_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.merge(predict_BI_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.rename(columns={"consumption": "BI-LSTM"})

predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'] = pd.to_datetime(predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['datetime_str'] = predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.merge(predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.rename(columns={"consumption": "CNN-BI-LSTM"})

predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'] = pd.to_datetime(predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['datetime_str'] = predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.merge(predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.rename(columns={"consumption": "CNN-LSTM"})

predict_XGBoost_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'] = pd.to_datetime(predict_XGBoost_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_XGBoost_df_DEKENERG_ZONE2_S_PAMURENE_df['datetime_str'] = predict_XGBoost_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.merge(predict_XGBoost_df_DEKENERG_ZONE2_S_PAMURENE_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.rename(columns={"consumption": "XGBoost"})

predict_AutoNHITS_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'] = pd.to_datetime(predict_AutoNHITS_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_AutoNHITS_df_DEKENERG_ZONE2_S_PAMURENE_df['datetime_str'] = predict_AutoNHITS_df_DEKENERG_ZONE2_S_PAMURENE_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.merge(predict_AutoNHITS_df_DEKENERG_ZONE2_S_PAMURENE_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.rename(columns={"consumption": "AutoNHITS"})


df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.drop('datetime_str', axis=1)
df_all_DEKENERG_ZONE2_S_PAMURENE = df_all_DEKENERG_ZONE2_S_PAMURENE.iloc[:(24*31)-1]


df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_DEKENERG_ZONE2_S_PEVRAOBL.copy()

df_all_DEKENERG_ZONE2_S_PEVRAOBL['datetime'] = pd.to_datetime(df_all_DEKENERG_ZONE2_S_PEVRAOBL['datetime'], format='%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PEVRAOBL['datetime_str'] = df_all_DEKENERG_ZONE2_S_PEVRAOBL['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

predict_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'] = pd.to_datetime(predict_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['datetime_str'] = predict_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.merge(predict_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.rename(columns={"consumption": "LSTM"})

predict_BI_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'] = pd.to_datetime(predict_BI_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_BI_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['datetime_str'] = predict_BI_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.merge(predict_BI_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.rename(columns={"consumption": "BI-LSTM"})

predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'] = pd.to_datetime(predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['datetime_str'] = predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.merge(predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.rename(columns={"consumption": "CNN-BI-LSTM"})

predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'] = pd.to_datetime(predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['datetime_str'] = predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.merge(predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.rename(columns={"consumption": "CNN-LSTM"})

predict_XGBoost_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'] = pd.to_datetime(predict_XGBoost_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_XGBoost_df_DEKENERG_ZONE2_S_PEVRAOBL_df['datetime_str'] = predict_XGBoost_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.merge(predict_XGBoost_df_DEKENERG_ZONE2_S_PEVRAOBL_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.rename(columns={"consumption": "XGBoost"})


predict_AutoNHITS_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'] = pd.to_datetime(predict_AutoNHITS_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'], format='%Y-%m-%dT%H:%M:%S')
predict_AutoNHITS_df_DEKENERG_ZONE2_S_PEVRAOBL_df['datetime_str'] = predict_AutoNHITS_df_DEKENERG_ZONE2_S_PEVRAOBL_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.merge(predict_AutoNHITS_df_DEKENERG_ZONE2_S_PEVRAOBL_df[['datetime_str', 'consumption']], on='datetime_str', how='left')
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.rename(columns={"consumption": "AutoNHITS"})



df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.drop('datetime_str', axis=1)
df_all_DEKENERG_ZONE2_S_PEVRAOBL = df_all_DEKENERG_ZONE2_S_PEVRAOBL.iloc[:(24*31)-1]







st.write(f"# Glossary")

st.markdown("## Отчет подготовлен командой [Horizon TSD](https://time-horizon.ru)")


st.markdown("""
### **Сравнение прогнозов АТС**  
 Анализ точности прогноза по колонке `VC_ППП` (прогнозные значения) в сравнении с фактическими значениями `VC_факт` из исходных данных.  

#### **Обоснование выбора колонки:**  
 Согласно документации к данным, **`VC_ППП`** — это объем электрической энергии, включенный ГП (ЭСК, ЭСО) в объемы планового почасового потребления. Поэтому данная колонка была использована в качестве эталонного прогноза для сравнения.  

 *Источник данных:* [АТС Энерго](https://www.atsenergo.ru/results/market/mtncz) (период: с 2011 года по июнь 2016 года).  
 *Примечание:* После июня 2016 года данные по `VC_факт` и `VC_ППП` в отчетах отсутствуют.  

### **Прогноз Horizon TSD**  
 - Представление модели прогнозирования команды  
 - Метрики качества прогноза  
 - Сравнение с прогнозом АТС и демонстрация улучшений  

### **Экспериментальные данные**  
Анализ на двух тестовых выборках:  
 - ОАО 'ДЭК' ГТП: **PAMURENE**  
 - ОАО 'ДЭК' ГТП: **PEVRAOBL**  

### **Краткие выводы:**  
 ##### ✅ Все представленные модели показали улучшение точности от 4 до 30% по сравнению с VC_ППП на обоих датасетах
""")


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
                                         line=dict(color="orange")), row=1, col=1)

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

def our_forecast_graph(df, df_our, name):
    df_our = df_our.rename(columns={"Datetime": "datetime", "consumption": "predict"})

    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df_our['datetime'] = pd.to_datetime(df_our['datetime'], format='%Y-%m-%dT%H:%M:%S')
    df['datetime_str'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_our['datetime_str'] = df_our['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.merge(df_our[['datetime_str', 'predict']], on='datetime_str', how='left')
    df = df.drop('datetime_str', axis=1)

    df_to_eval = df.iloc[:(24*31)-1]

    y_true = df_to_eval["VC_факт"]
    y_pred = df_to_eval["predict"]

    rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)

    title = f"{name} прогноз"

    fig_consumption = make_subplots(rows=1, cols=1, subplot_titles=[title])

    fig_consumption.add_trace(
        go.Scatter(x=df_to_eval["datetime"], y=df_to_eval["VC_факт"], mode="lines", name="VC_факт", line=dict(color="blue")), row=1,
        col=1)
    fig_consumption.add_trace(go.Scatter(x=df_to_eval["datetime"], y=df_to_eval["VC_ППП"], mode="lines", name="VC_ППП",
                                         line=dict(color="orange")), row=1, col=1)

    fig_consumption.add_trace(go.Scatter(x=df_to_eval["datetime"], y=df_to_eval["predict"], mode="lines", name=f"{name} predict",
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
            name=f" R = {round(r2, 3)}"
        )
    )

    fig_consumption.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name=f" RMSE = {round(rmse, 3)}"
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

def our_metrix_visual(rmse, r2, mae, mape, wmape, our_rmse, our_r2, our_mae, our_mape, our_wmape, name):

    rmse_improve = improve_calc(rmse, our_rmse)
    r2_improve = improve_calc(r2, our_r2)
    mae_improve = improve_calc(mae, our_mae)
    mape_improve = improve_calc(mape, our_mape)
    wmape_improve = improve_calc(wmape, our_wmape)



    st.write(f'## Метрики {name} HORIZON прогноза')
    cols = st.columns(5)

    cols[0].write(f'##### АТС MAPE = {mape}%')
    with cols[0].container(height=130):
        st.metric("HORIZON MAPE", f"{our_mape} %", f"{mape_improve}%")

    cols[1].write(f'##### АТС R2 = {r2}')
    with cols[1].container(height=130):
        st.metric("HORIZON R2", f"{our_r2}", f"{r2_improve*(-1)}")

    cols[2].write(f'##### АТС RMSE = {rmse}')
    with cols[2].container(height=130):
        st.metric("HORIZON RMSE", f"{our_rmse}", f"{rmse_improve}")

    cols[3].write(f'##### АТС MAE = {mae}%')
    with cols[3].container(height=130):
        st.metric("HORIZON MAE", f"{our_mae} %", f"{mae_improve}%")

    cols[4].write(f'##### АТС WMAPE = {wmape}%')
    with cols[4].container(height=130):
        st.metric("HORIZON WMAPE", f"{our_wmape} %", f"{wmape_improve}%")

# @check_authentication
async def main():

    st.markdown('---')

    st.title('ОАО "ДЭК" ГТП: PAMURENE | DEKENERG ZONE2 S PAMURENE')

    rmse, r2, mae, mape, wmape = atc_forecast_graph(df_DEKENERG_ZONE2_S_PAMURENE)

    ats_metrix_visual(rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape)
    # st.dataframe(df_DEKENERG_ZONE2_S_PAMURENE, use_container_width=True)

    name = 'XGBoost'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PAMURENE,
        df_our=predict_XGBoost_df_DEKENERG_ZONE2_S_PAMURENE_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    name = 'LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PAMURENE,
        df_our=predict_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
        )

    name = 'Bi-LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PAMURENE,
        df_our=predict_BI_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    name = 'CNN-Bi-LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PAMURENE,
        df_our=predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    name = 'CNN-LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PAMURENE,
        df_our=predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PAMURENE_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )
    st.write(f'### Результирующая таблица прогнозов по ОАО "ДЭК" ГТП: PAMURENE')
    st.dataframe(df_all_DEKENERG_ZONE2_S_PAMURENE, use_container_width=True)

    st.markdown('---')
    st.divider()
    st.divider()
    st.title('ОАО "ДЭК" ГТП: PEVRAOBL | DEKENERG ZONE2 S PEVRAOBL')
    rmse, r2, mae, mape, wmape = atc_forecast_graph(df_DEKENERG_ZONE2_S_PEVRAOBL)
    ats_metrix_visual(rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape)

    name = 'XGBoost'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PEVRAOBL,
        df_our=predict_XGBoost_df_DEKENERG_ZONE2_S_PEVRAOBL_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    name = 'CNN-LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PEVRAOBL,
        df_our=predict_CNN_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    name = 'Bi-LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PEVRAOBL,
        df_our=predict_BI_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    name = 'LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PEVRAOBL,
        df_our=predict_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    name = 'CNN-Bi-LSTM'
    st.title(name)
    our_rmse, our_r2, our_mae, our_mape, our_wmape = our_forecast_graph(
        df=df_DEKENERG_ZONE2_S_PEVRAOBL,
        df_our=predict_CNN_Bi_LSTM_df_DEKENERG_ZONE2_S_PEVRAOBL_df,
        name=name
    )
    our_metrix_visual(
        rmse=rmse, r2=r2, mae=mae, mape=mape, wmape=wmape,
        our_rmse=our_rmse, our_r2=our_r2, our_mae=our_mae, our_mape=our_mape, our_wmape=our_wmape,
        name=name
    )

    st.write(f'### Результирующая таблица прогнозов по ОАО "ДЭК" ГТП: PEVRAOBL')
    st.dataframe(df_all_DEKENERG_ZONE2_S_PEVRAOBL, use_container_width=True)

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



