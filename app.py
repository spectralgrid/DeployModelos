# Bibliotecas
from shiny import reactive
from shiny.express import input, render, ui
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import plotnine as p9
from pathlib import Path


# Dados
# url_base = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO="
# url_p = f"{url_base}'EIA366_PBRENT366')"
# url_c = f"{url_base}'GM366_EREURO366')"
# 
# dados_petroleo = pd.DataFrame.from_records(pd.read_json(url_p).value.values)
# dados_cambio = pd.DataFrame.from_records(pd.read_json(url_c).value.values)
# 
# petroleo = pd.DataFrame(
#   data = {"brent": dados_petroleo.VALVALOR.astype(float).apply(np.log).values},
#   index = pd.to_datetime(dados_petroleo.VALDATA, utc = True).rename("data")
#   )
# cambio = pd.DataFrame(
#   data = {"eurusd": dados_cambio.VALVALOR.astype(float).values},
#   index = pd.to_datetime(dados_cambio.VALDATA, utc = True).rename("data")
#   )
# 
# dados = petroleo.join(cambio, how = "outer")
# dados.insert(0, "data", pd.to_datetime(dados.index.date))
# dados.to_csv("app/dados.csv", index= False)

def ler_csv():
    infile = Path(__file__).parent / "dados.csv"
    df_dados = pd.read_csv(
      infile, 
      converters = {"data": lambda x: pd.to_datetime(x)}
      )
    return df_dados
  
dados = ler_csv()
dados.index = dados.data

# Inputs
ui.page_opts(title = ui.strong("Preço do Petróleo"), fillable = True)

with ui.sidebar(width = 300):
    ui.input_date(
        id = "periodo",
        label = "Filtrar data inicial:",
        value = (dados.index.max() - pd.offsets.BusinessDay(n = 90)).date(),
        min = dados.index.min().date(),
        max = dados.index.max().date(),
        language = "pt-BR"
        )
    ui.input_select(
        id = "modelo",
        label = "Modelo de previsão:",
        choices = ["Regressão linear", "SARIMAX", "VAR"],
        selected = "VAR"
        )
    ui.input_numeric(
        id = "horizonte", 
        label = "Horizonte de previsão:",
        value = 7,
        min = 1,
        max = 30,
        step = 1
        )
    ui.input_numeric(
        id = "intervalo", 
        label = "Intervalo de confiança:",
        value = 80,
        min = 1,
        max = 100,
        step = 1
        )
    ui.markdown("[Análise Macro](https://analisemacro.com.br/)")


# Outputs
with ui.navset_card_underline(title = "Previsão"):
    with ui.nav_panel("Gráfico"):
        @render.plot
        def grafico():

          df_previsao = (
            previsao()
            .assign(
              brent = lambda x: x["Previsão"],
              data = lambda x: pd.to_datetime(x["Período"])
              )
            )
          
          data_ini = input.periodo().strftime("%Y-%m-%d")
          df_filtrado = (
            dados
            .query("data >= @data_ini")
            .assign(brent = lambda x: np.exp(x.brent))
            )
            
          df_grafico = pd.concat([df_filtrado, df_previsao])

          from datetime import date

          def weekinmonth(dates):
              firstday_in_month = dates - pd.to_timedelta(dates.day - 1, unit='d')
              return (dates.day-1 + firstday_in_month.weekday()) // 7 + 1
          
          def custom_date_format2(breaks):
              res = []
              for x in breaks:
                  # First day of the year
                  if x.month == 1 and weekinmonth(x) == 1:
                      fmt = "%Y"
                  # Every other month
                  elif x.month != 1 and weekinmonth(x) == 1:
                      fmt = "%b"
                  else:
                      fmt = "%d"

                  res.append(date.strftime(x, fmt))

              return res
            
          return (
            p9.ggplot(df_grafico) +
            p9.aes(x = "data", y = "Previsão") +
            p9.geom_line(mapping = p9.aes(y = "brent"), size = 1.5) +
            p9.geom_line(size = 2, color = "blue") +
            p9.geom_ribbon(
              mapping = p9.aes(ymin = "Intervalo Inferior", ymax = "Intervalo Superior"),
              alpha = 0.25,
              fill = "blue"
              ) +
            p9.scale_x_date(
                date_breaks = "1 week" if df_grafico.shape[0] < 120 else "1 month", 
                labels = custom_date_format2, 
                date_minor_breaks = "1 week"
                ) +
            p9.theme_gray(base_size = 12) +
            p9.theme(axis_text_x = p9.element_text(angle = 90)) +
            p9.labs(y = "US$ / barril Brent", x = "")
          )
          

    with ui.nav_panel("Tabela"):
        @render.data_frame
        def tabela():
            return render.DataGrid(previsao(), editable = False, width = "100%")


# Servidor
@reactive.calc
def previsao():
  
    modelo = input.modelo()
    h = input.horizonte()
    ic = 1 - input.intervalo()/100

    df_treino = dados.dropna(subset = "brent").query("data > '2014-01-01'")
    df_ex = dados.query("data > @df_treino.data.max()")
    
    index_cenario = pd.date_range(
        start = df_ex.index.max() + pd.DateOffset(days = 1), 
        end = df_ex.index.max() + pd.DateOffset(days = h - df_ex.shape[0]), 
        freq = "D"
        )
    df_cenario = pd.DataFrame(
      data = {
        "eurusd": pd.Series(
          [df_ex.eurusd.iloc[-1]] * (h - df_ex.shape[0])
          ).values,
        "data": pd.to_datetime(index_cenario.date),
        "brent": np.nan
        },
      index = index_cenario
      )
    df_previsao = pd.concat([df_ex, df_cenario])
    
    
    if modelo == "Regressão linear":
        mod = smf.ols(formula = "brent ~ eurusd", data = df_treino).fit()
        prev = mod.get_prediction(df_previsao)
        df_prev = pd.DataFrame(
          data = {
            "Período": pd.to_datetime(df_previsao.index.date).astype(str),
            "Intervalo Inferior": np.exp(prev.conf_int(alpha = ic)[:,0]),
            "Previsão": np.exp(prev.predicted),
            "Intervalo Superior": np.exp(prev.conf_int(alpha = ic)[:,1]),
          }
        )
    elif modelo == "SARIMAX":
        mod = sm.tsa.statespace.SARIMAX(
          endog = df_treino.dropna()["brent"],
          exog = df_treino.dropna()["eurusd"],
          order = (1, 0, 0),
          seasonal_order = (0, 0, 1, 12),
          trend = "ct"
          ).fit()
        prev = mod.get_forecast(
          steps = df_previsao.shape[0],
          exog = df_previsao.eurusd
          )
        df_prev = pd.DataFrame(
          data = {
            "Período": pd.to_datetime(df_previsao.index.date).astype(str),
            "Intervalo Inferior": np.exp(prev.conf_int(alpha = ic)["lower brent"]),
            "Previsão": np.exp(prev.predicted_mean.values),
            "Intervalo Superior": np.exp(prev.conf_int(alpha = ic)["upper brent"]),
          }
        )
    else:
        mod = VAR(df_treino.drop(labels = "data", axis = "columns").dropna()).fit(
          maxlags = 15,
          trend = "ct"
          )
        prev = mod.forecast_interval(
          y = df_treino.drop(labels = "data", axis = "columns").dropna().values[-mod.k_ar:],
          steps = df_previsao.shape[0],
          alpha = ic
          )
        df_prev = pd.DataFrame(
          data = {
            "Período": pd.to_datetime(df_previsao.index.date).astype(str),
            "Intervalo Inferior": np.exp(prev[1][:,0]),
            "Previsão": np.exp(prev[0][:,0]),
            "Intervalo Superior": np.exp(prev[2][:,0]),
          }
        )
      
      
    return df_prev
