# %%
# librerias
import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy import stats
from scipy.stats import linregress

import matplotlib.pyplot as plt

import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

key = "7d99d108d429f09277ea5ce381cf96e8"

# %%
# funciones
def to_returns(prices: pd.DataFrame, method="log"):
    """Convierte precios a retornos diarios."""
    if method == "log":
        return np.log(prices).diff().dropna()
    else:
        return prices.pct_change().dropna()

def annualize_mean(r, periods_per_year=252):
    return r.mean() * periods_per_year

def annualize_vol(r, periods_per_year=252):
    return r.std() * np.sqrt(periods_per_year)

def sharpe_ratio(r, rf=0.0, periods_per_year=252):
    # rf en misma frecuencia que r (si rf anual, conviértelo antes)
    excess = r - rf
    return annualize_mean(excess, periods_per_year) / annualize_vol(r, periods_per_year)

def random_weights(k, n_portfolios=5000):
    W = np.random.dirichlet(np.ones(k), size=n_portfolios)
    return W

def portfolio_stats(rets: pd.DataFrame, W: np.ndarray, periods_per_year=252):
    mu = rets.mean().values
    Sigma = rets.cov().values
    port_mu = W @ mu
    port_var = np.einsum('ij,jk,ik->i', W, Sigma, W)
    port_vol = np.sqrt(port_var)
    return (port_mu * periods_per_year, port_vol * np.sqrt(periods_per_year))

def summarize_model(m):
    out = pd.DataFrame({
        "coef": m.params,
        "pval": m.pvalues
    })
    return out, m.rsquared

# %% [markdown]
# ## El mercado NO paga por riesgo diversificable
# 
# Vamos a crear un “mundo” con 3 activos correlacionados y ver qué le pasa al riesgo al diversificar.

# %%
np.random.seed(7)

n = 1500  # ~6 años de días hábiles
mu = np.array([0.0004, 0.00035, 0.0003])  # medias diarias
cov = np.array([
    [0.00018, 0.00012, 0.00010],
    [0.00012, 0.00020, 0.00011],
    [0.00010, 0.00011, 0.00016]
])

R = np.random.multivariate_normal(mu, cov, size=n)
rets = pd.DataFrame(R, columns=["A", "B", "C"])

print(rets.head(3))
print()
weights = np.array([1/3, 1/3, 1/3])
port = rets @ weights

print("Vol anual A:", annualize_vol(rets["A"]))
print("Vol anual B:", annualize_vol(rets["B"]))
print("Vol anual C:", annualize_vol(rets["C"]))
print("Vol anual Port(1/3):", annualize_vol(port))

# %% [markdown]
# # 2 Markowitz frontera eficiente

# %%
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Matriz A:\n", A)
print("Matriz B:\n", B)

# Producto matricial (A * B)
resultado = np.einsum('ij,jk->ik', A, B)
# Resultado: [[19, 22], [43, 50]]
print("Producto matricial A * B:\n", resultado)

# %%
# Define concentration parameters (e.g., 3 categories)
alpha = [10, 5, 1] 

# Generate a single sample (a vector of 3 probabilities summing to 1)
sample = np.random.dirichlet(alpha)
print("Single Sample:", sample)
print("Sum:", np.sum(sample))

# Generate 5 samples
samples = np.random.dirichlet(alpha, size=5)
for i, s in enumerate(samples):
    print(f"Sample {i+1}: {s}, Sum: {np.sum(s)}")
print("\n5 Samples:\n", samples)


# %%
W = random_weights(3, n_portfolios=8000)
mu_a, vol_a = portfolio_stats(rets, W)

# Grafico portafolios
plt.figure()
plt.scatter(vol_a, mu_a, s=6)
plt.xlabel("Volatilidad anual")
plt.ylabel("Retorno anual esperado (aprox.)")
plt.title("Portafolios (Markowitz por simulación)")
plt.show()

# %%
vol_a

# %%
index = (vol_a == 0.19376)
print("Portafolio con volatilidad ~0.19:", mu_a[index], vol_a[index], W[index])

# %%
weights_real = np.array([0.79679676, 0.09852811, 0.10467514])
portfolio_stats(rets, W = weights_real.reshape(1, -1), periods_per_year=252)

# %%
rf = 0.06
weights_real = np.array([0.79679676, 0.09852811, 0.10467514])  # Ejemplo de pesos reales
rm, vol_m = portfolio_stats(rets, W = weights_real.reshape(1, -1), periods_per_year=252)

vol_grid = np.linspace(0.01, max(vol_a), 1000)
cml = rf + (rm - rf) * (vol_grid / vol_m)
plt.figure()
plt.scatter(vol_a, mu_a, s=6, label="Portafolios simulados")
plt.scatter(vol_m, rm, s=50, color='green', label="Portafolio de mercado")
# plt.plot(vol_grid, cml, color='red', label="CML")
plt.xlabel("Volatilidad anual")
plt.ylabel("Retorno anual esperado (aprox.)")
plt.title("Portafolios y CML")
plt.legend()
plt.show()


# %%
W = random_weights(3, n_portfolios=8000)
mu_a, vol_a = portfolio_stats(rets, W)

# agregando rf y rm
rf = 0.02
weights_real = np.array([0.5, 0.3, 0.2])  # Ejemplo de pesos reales
rm, vol_m = portfolio_stats(rets, W = weights_real.reshape(1, -1), periods_per_year=252)


vol_p = 0.19
rp = rf + (rm - rf) * (vol_p / vol_m)

# Grafico portafolios
plt.figure()
plt.scatter(vol_a, mu_a, s=6)
plt
plt.xlabel("Volatilidad anual")
plt.ylabel("Retorno anual esperado (aprox.)")
plt.title("Portafolios (Markowitz por simulación)")
plt.show()

# %% [markdown]
# # 3 CAPM: beta, alpha y la trampa clásica

# %% [markdown]
# ## 3.1 Simulamos un mercado y un activo

# %%
np.random.seed(12)

n = 1800
mkt = np.random.normal(0.00035, 0.011, size=n)  # retorno mercado diario
rf = 0.00008  # tasa libre riesgo diaria

true_beta = 1.3
true_alpha = 0.00002  # muy pequeño
eps = np.random.normal(0, 0.014, size=n)

asset = true_alpha + true_beta * (mkt - rf) + eps

df = pd.DataFrame({
    "MKT": mkt,
    "RF": rf,
    "ASSET": asset
})

df["MKT_EXCESS"] = df["MKT"] - df["RF"]
df["ASSET_EXCESS"] = df["ASSET"] - df["RF"]
df.head(8)

# %% [markdown]
# ## 3.2 Estimación CAPM (OLS)

# %%
X = sm.add_constant(df["MKT_EXCESS"])
y = df["ASSET_EXCESS"]

model = sm.OLS(y, X).fit()
print(model.summary())

# %% [markdown]
# Interpretación guiada (lo importante):
# 
# * $\beta$ (beta): sensibilidad al mercado.
# * $\alpha$ (alpha): “retorno extra” no explicado por mercado (a veces suerte disfrazada).
# * $Rˆ2$: qué tanto explica el mercado el activo.

# %% [markdown]
# ## 3.3 El test de “¿alpha real o humo?”

# %% [markdown]
# Alpha pequeño + p-value alto = probablemente nada.

# %%
alpha = model.params["const"]
beta = model.params["MKT_EXCESS"]
p_alpha = model.pvalues["const"]

print("alpha diaria:", alpha)
print("beta:", beta)
print("p-value alpha:", p_alpha)

# %% [markdown]
# ## 3.4 SML (línea de mercado de valores) en versión práctica

# %% [markdown]
# CAPM en expectativas:
# 
# $$ E[R_i] = R_f + \beta_i(E[R_m]-R_f) $$

# %%
# estimamos primas anuales con nuesrtras medias muestrales
Erm = annualize_mean(df["MKT"])
Erf = annualize_mean(df["RF"])
capm_return = Erf + beta * (Erm - Erf)

print("E[Rm] anual:", Erm)
print("Rf anual:", Erf)
print("E[Ri] CAPM anual:", capm_return)
print("E[Ri] muestral anual:", annualize_mean(df["ASSET"]))

# %% [markdown]
# 	• Si el CAPM “no cuadra” no significa que no sirva el modelo.
# 	• Significa que revises: ventana temporal, estabilidad, regime change, datos, etc.

# %% [markdown]
# ## Data Real

# %%
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
rets = to_returns(data)
print(rets.head(3))

weights = np.array([1/3, 1/3, 1/3])
port = rets @ weights
print()
print(f"Vol anual {tickers[0]}:", annualize_vol(rets[tickers[0]]))
print(f"Vol anual {tickers[1]}:", annualize_vol(rets[tickers[1]]))
print(f"Vol anual {tickers[2]}:", annualize_vol(rets[tickers[2]]))
print(f"Vol anual Port(1/3):", annualize_vol(port))

# %%
print(f"Mean anual {tickers[0]}:", annualize_mean(rets[tickers[0]]))
print(f"Mean anual {tickers[1]}:", annualize_mean(rets[tickers[1]]))
print(f"Mean anual {tickers[2]}:", annualize_mean(rets[tickers[2]]))
print(f"Mean anual Port(1/3):", annualize_mean(port))

# %%
print(f"Sharpe anual {tickers[0]}:", annualize_mean(rets[tickers[0]]) / annualize_vol(rets[tickers[0]]))
print(f"Sharpe anual {tickers[1]}:", annualize_mean(rets[tickers[1]]) / annualize_vol(rets[tickers[1]]))
print(f"Sharpe anual {tickers[2]}:", annualize_mean(rets[tickers[2]]) / annualize_vol(rets[tickers[2]]))
print(f"Sharpe anual Port(1/3):", annualize_mean(port) / annualize_vol(port))

# %%
W = random_weights(3, n_portfolios=8000)
mu_a, vol_a = portfolio_stats(rets, W)

plt.figure()
plt.scatter(vol_a, mu_a, s=6)
plt.xlabel("Volatilidad anual")
plt.ylabel("Retorno anual esperado (aprox.)")
plt.title("Portafolios (Markowitz por simulación)")
plt.show()

# %%
tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
rets = to_returns(data)

benchmark = "SPY"
data_bench = yf.download(benchmark, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
rets_bench = to_returns(data_bench)

# %%
returns_benchmark = rets["SPY"].values
slope, intercept, r_value, p_value, std_err  = linregress(returns_benchmark, rets["AAPL"].values)
alpha = np.round(intercept, 4)
beta = np.round(slope, 4)
null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
r_squared = np.round(r_value**2, 4)
correlation = r_value
predictor_linreg = alpha + beta*returns_benchmark

print(f"AAPL - Beta: {beta}, Alpha diaria: {alpha}, p-value alpha: {p_value:.4f}, correlation: {correlation:.4f}, R^2: {r_squared:.4f}")


# %%
str_title = 'Scatterplot of returns' + '\n' + tickers[0]
plt.figure()
plt.title(str_title)
plt.scatter(returns_benchmark, rets["AAPL"].values)
plt.plot(returns_benchmark, predictor_linreg, color='green')
plt.ylabel(tickers[0])
plt.xlabel(tickers[-1])
plt.grid()
plt.show()

# %%
X = sm.add_constant(returns_benchmark)
y = rets["AAPL"].values

model = sm.OLS(y, X).fit()
print(model.summary())

# %%
returns_benchmark = rets["SPY"].values
for ticker in tickers[:-1]:  # excluimos SPY
    returns_asset = rets[ticker].values
    slope, intercept, r_value, p_value, std_err = linregress(returns_benchmark, returns_asset)
    print(f"{ticker} - Beta: {slope:.4f}, Alpha diaria: {intercept:.6f}, p-value alpha: {p_value:.4f}, correlation: {r_value:.4f}, R^2: {r_value**2:.4f}")


# %% [markdown]
# ## 3.6 Portafolio de betas maximizando con sharpe ratio
# 
# Supongamos que tenemos un portafolio $P$ conformado por $n$ activos. La beta de dicho portafolio estará determinada por
# 
# $$\beta_P = \sum_{i=1}^n w_i * \beta_i$$
# 
# donde $\beta_i$ es la beta del activo $i$ en el portafolio y $w_i$ es el peso del activo $i$ dentro del portafolio.
# 
# El Sharpe ratio se define como
# 
# $$\dfrac{r_p-r_f}{\sigma_p}$$

# %%
rets.head(3)

# %%
# function compute beta
def compute_beta(returns_asset, returns_benchmark):
    beta, alpha, r_value, p_value, std_err = linregress(returns_benchmark, returns_asset)
    return beta, alpha

compute_beta(rets["AAPL"].values, rets["SPY"].values)

# %%
# function beta portfolio
    
def compute_portfolio_beta(weights, betas):
    return np.dot(weights, betas)

# %%
betas_portfolio = []
for ticker in tickers[:-1]:  # excluimos SPY
    # print(f"Calculando beta para {ticker}...")
    beta ,_ = compute_beta(rets[ticker].values, rets['SPY'].values)
    betas_portfolio.append(beta)
    print(f"Beta {ticker}: {beta:.4f}")

# %%
print("\nBetas del portafolio:")
compute_portfolio_beta(weights, betas_portfolio)

# %%
betas_portfolio


# %%
W = random_weights(3, n_portfolios=8000)
betas_portafolios = W @ np.array(betas_portfolio)
mu_a, vol_a = portfolio_stats(rets.iloc[:, :-1], W)

plt.scatter(vol_a, mu_a, c=betas_portafolios, cmap='viridis')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Beta')
plt.show()

# %%
plt.scatter(betas_portafolios, mu_a, )
plt.xlabel('Beta')
plt.ylabel('Expected Return')
plt.show()

# %% [markdown]
# ## 3.7 CAPM Cobertura

# %%
tickers = ["NVDA", "SPY"]
data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
rets = to_returns(data)
rets.head(3)

beta, alpha = compute_beta(rets["NVDA"].values, rets["SPY"].values)
print(f"Beta NVDA: {beta:.4f}, Alpha diaria: {alpha:.6f}")

# %% [markdown]
# Supongamos que tenemos invertidos $100,000 ´SPY´ y queremos cubrir esta posición. Si lo quisieramos hacer de manera "natural" con ´NVDA´ tendríamos que ir cortos con $100,000 en ´NVDA´, sin embargo esto no garantiza una cobertura robusta, ya que si ´SPY´ sube 1\% ganaríamos $1,000 pero de acuerdo al CAPM con ´NVDA´ perderíamos $1,800 aprox. (**Cobertura delta neutral**)
# Si baja el ´SPY´ 1\% perderíamos $1,000 pero ganaríamos $1,800 aprox. con ´NVDA´.
# 
# Para robustecer esta cobertura podemos considerar las betas para equilibar el riesgo y el monto a invertir.

# %%
tickers = ["NVDA", "MSFT" , "META", "SPY"]
data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
rets = to_returns(data)
rets.head(3)

beta, alpha = compute_beta(rets["NVDA"].values, rets["SPY"].values)
print(f"Beta NVDA: {beta:.4f}, Alpha diaria: {alpha:.6f}")

beta, alpha = compute_beta(rets["MSFT"].values, rets["SPY"].values)
print(f"Beta MSFT: {beta:.4f}, Alpha diaria: {alpha:.6f}")

beta, alpha = compute_beta(rets["META"].values, rets["SPY"].values)
print(f"Beta META: {beta:.4f}, Alpha diaria: {alpha:.6f}")

# %% [markdown]
# Imaginemos que estamos invertidos con $100,000 en ´META´ (activo $S_0$) y queremos cubrir la posición con ´NVDA´ (activo $S_1$) y ´MSFT´ (activo $S_2$). Entonces podemos establecer el siguiente sistema de ecuaciones para encontrar el monto a invertir en ´NVDA´ y ´MSFT´:

# %% [markdown]
# $$S_0 + S_1 + S_2 = 0, \text{cobertura delta neutral} $$
# 
# $$\beta_0 S_0 + \beta_1 S_1 + \beta_2 S_2 = 0, \text{cobertura beta neutral} $$

# %% [markdown]
# $$
# \begin{pmatrix}
#   1 & 1  \\
#   \beta_1 & \beta_2 \\
# \end{pmatrix}
# \begin{pmatrix} S_1\\ S_2\\ \end{pmatrix} = \begin{pmatrix} -S_0\\ -\beta_0 S_0\\ \end{pmatrix}
# $$
# 
# $$ \therefore
# \begin{pmatrix} S_1\\ S_2\\ \end{pmatrix} = \begin{pmatrix}
#   1 & 1  \\
#   \beta_1 & \beta_2 \\
# \end{pmatrix}^{-1}
# \begin{pmatrix} -S_0\\ -\beta_0 S_0\\ \end{pmatrix}
# $$

# %%
ric = 'META' #'REP.MC' #'ba' #BBVA.MC
benchmark = 'SPY'  #  'SPY'   '^STOXX'
hedge_rics = ['NVDA', 'MSFT']
delta = 10  #10 M usd
tickers = [ric] + hedge_rics + [benchmark]

data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
rets = to_returns(data)

# %%
beta = compute_beta(rets[ric].values, rets[benchmark].values)[0]
beta_usd = beta * delta
print(f"Beta {ric}: {beta:.4f}, Beta en USD: {beta_usd:.2f}")

# %%
betas = [compute_beta(rets[hedge_ric].values, rets[benchmark].values)[0] \
                        for hedge_ric in hedge_rics]
print("Betas hedge:", betas)

# %%
betas = np.asarray(betas).reshape([len(hedge_rics),1])

dataframe = pd.DataFrame()
dataframe['ric'] = hedge_rics
dataframe['beta'] = betas

print('------')
print('Input portfolio:')
print('Delta mnUSD for ' + ric + ' is ' + str(delta))
print('Beta for ' + ric + ' vs ' + benchmark + ' is ' + str(beta))
print('Beta mnUSD for ' + ric + ' vs ' + benchmark + ' is ' + str(beta_usd))
print('------')
print('Input hedges:')
for n in range(dataframe.shape[0]):
    print('Beta for hedge[' + str(n) + '] = ' + dataframe['ric'][n] \
          + ' vs ' + benchmark + ' is ' + str(dataframe['beta'][n]))

# %%
print(dataframe)
print(dataframe.shape)
np.ones([3,2])

# %% [markdown]
# $$ \therefore
# \begin{pmatrix} S_1\\ S_2\\ \end{pmatrix} = \begin{pmatrix}
#   1 & 1  \\
#   \beta_1 & \beta_2 \\
# \end{pmatrix}^{-1}
# \begin{pmatrix} -S_0\\ -\beta_0 S_0\\ \end{pmatrix}
# $$

# %%
size = dataframe.shape[0] # numero de activos para la cobertura
deltas = np.ones([size,1]) # vector columna de deltas (1 por cada activo de la cobertura)
targets = -np.array([[delta],[beta_usd]]) # vector a cubrir (delta y beta en USD)
mtx = np.transpose(np.column_stack((deltas, betas))) # matrix a invertir
optimal_hedge = np.linalg.inv(mtx).dot(targets) # solucion del sistema de ecuaciones para encontrar los pesos de la cobertura
dataframe['delta'] = optimal_hedge # cantidad a invertir en cada activo de la cobertura
dataframe['beta_usd'] = betas*optimal_hedge # beta en USD para cada activo de la cobertura
hedge_delta = np.sum(dataframe['delta'])
hedge_beta_usd = np.sum(dataframe['beta_usd']) #np.transpose(betas).dot(optimal_hedge).item()

# %%
print('------')
print(str(ric))
# print('Optimisation result | ' + optimisation_type + ':')
print('Optimisation result :')
print('------')
print('Delta: ' + str(delta))
print('Beta USD: ' + str(beta_usd))
print('------')
print('Hedge delta: ' + str(hedge_delta))
print('Hedge beta USD: ' + str(hedge_beta_usd))
print('------')
print('Betas for the hedge:')
print(betas)
print('------')
print('Hegde rics:')
print(hedge_rics)
print('Optimal hedge:')
print(optimal_hedge)

# %%
def cobertura_capm(ric, benchmark, hedge_rics, delta):
    tickers = [ric] + hedge_rics + [benchmark]
    data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
    rets = to_returns(data)
    
    beta = compute_beta(rets[ric].values, rets[benchmark].values)[0]
    beta_usd = beta * delta
    print(f"Beta {ric}: {beta:.4f}, Beta en USD: {beta_usd:.2f}, \n")
    
    betas = [compute_beta(rets[hedge_ric].values, rets[benchmark].values)[0] for hedge_ric in hedge_rics]
    print("Betas hedge:", betas, "\n")
    betas = np.asarray(betas).reshape([len(hedge_rics),1])
    dataframe = pd.DataFrame()
    dataframe['ric'] = hedge_rics
    dataframe['beta'] = betas
    size = dataframe.shape[0] # numero de activos para la cobertura
    deltas = np.ones([size,1]) # vector columna de deltas (1 por cada activo de la cobertura)
    targets = -np.array([[delta],[beta_usd]]) # vector a cubrir (delta
    # y beta en USD)
    mtx = np.transpose(np.column_stack((deltas, betas))) # matrix a invertir
    optimal_hedge = np.linalg.inv(mtx).dot(targets) # solucion del sistema de ecuaciones para encontrar los pesos de la cobertura
    dataframe['delta'] = optimal_hedge # cantidad a invertir en cada activo de la cobertura
    dataframe['beta_usd'] = betas*optimal_hedge # beta en USD para cada activo de la cobertura
    hedge_delta = np.sum(dataframe['delta'])
    hedge_beta_usd = np.sum(dataframe['beta_usd']) 
    return dataframe, hedge_delta, hedge_beta_usd

# %%
ric = 'REP.MC'  #'META' #'REP.MC' #'ba' #BBVA.MC # S0
benchmark = '^STOXX'  #  'SPY'   '^STOXX'
hedge_rics = ['SAN.MC', 'BBVA.MC'] # S1, S2
delta = 10  #10 M usd

dataframe, hedge_delta, hedge_beta_usd = cobertura_capm(ric, benchmark, hedge_rics, delta)
dataframe.head(len(hedge_rics))

# %%
ric = 'META' #'REP.MC' #'ba' #BBVA.MC
benchmark = 'SPY'  #  'SPY'   '^STOXX'
hedge_rics = ['NVDA', 'MSFT', 'JPM']
delta = 10  #10 M usd
tickers = [ric] + hedge_rics + [benchmark]

data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
rets = to_returns(data)

beta = compute_beta(rets[ric].values, rets[benchmark].values)[0]
beta_usd = beta * delta
print(f"Beta {ric}: {beta:.4f}, Beta en USD: {beta_usd:.2f}, \n")

betas = [compute_beta(rets[hedge_ric].values, rets[benchmark].values)[0] \
                        for hedge_ric in hedge_rics]
print("Betas hedge:", betas, "\n")

# %%
from scipy.optimize import minimize

def cost_function_beta_delta(x, delta, beta_usd, betas, epsilon=0.0):
    f_delta = (sum(x).item() + delta)**2
    y = x.T
    f_beta = (np.transpose(betas).dot(y).item() + beta_usd)**2
    f_penalty = epsilon * sum(x**2).item()
    f = f_delta + f_beta + f_penalty
    return f

epsilon=0.01
x = np.zeros([len(betas),])
args = (delta, beta_usd, betas, epsilon)
optimal_result = minimize(fun = cost_function_beta_delta,\
                          x0=x, args=args, method='BFGS')

optimal_hedge = optimal_result.x.tolist()

df = pd.DataFrame()

df['rics hedge'] = hedge_rics
df['beta'] = betas
df['delta'] = optimal_hedge
df['beta_usd'] = np.array(betas) *optimal_result.x
hedge_delta = np.sum(df['delta'])
hedge_beta_usd = np.sum(df['beta_usd'])

df.head(len(hedge_rics))

# %%
print('------')
# print('Optimisation result | ' + optimisation_type + ':')
print('Optimisation result :')
print('------')
print('Delta: ' + str(delta))
print('Beta USD: ' + str(beta_usd))
print('------')
print('Hedge delta: ' + str(hedge_delta))
print('Hedge beta USD: ' + str(hedge_beta_usd))
print('------')
print('Betas for the hedge:')
print(betas)
print('------')
print('Hegde rics:')
print(hedge_rics)

print('Optimal hedge:')
print(optimal_hedge)

# %% [markdown]
# # 4 Modelos multifactor: cuando una beta no alcanza

# %% [markdown]
# ## 4.1 Creamos factores “estilo”

# %% [markdown]
# Ejemplo típico: mercado + tamaño + valor + momentum (toy model).

# %%
np.random.seed(21)

n = 1800
mkt = np.random.normal(0.00035, 0.011, size=n)
smb = np.random.normal(0.00010, 0.006, size=n)   # size
hml = np.random.normal(0.00008, 0.006, size=n)   # value
mom = np.random.normal(0.00012, 0.007, size=n)   # momentum
rf  = np.full(n, 0.00008)

# Activo con exposición a varios factores
b = {"mkt": 1.1, "smb": 0.4, "hml": -0.2, "mom": 0.5}
alpha_true = 0.00001
eps = np.random.normal(0, 0.013, size=n)

asset = alpha_true + b["mkt"]*(mkt-rf) + b["smb"]*smb + b["hml"]*hml + b["mom"]*mom + eps

df2 = pd.DataFrame({
    "RF": rf,
    "MKT_EXCESS": mkt-rf,
    "SMB": smb,
    "HML": hml,
    "MOM": mom,
    "ASSET_EXCESS": asset-rf
})

print(df2.head(3))

# %%
df2.head(15)

# %%
# CAPM
X1 = sm.add_constant(df2[["MKT_EXCESS"]])
capm = sm.OLS(df2["ASSET_EXCESS"], X1).fit()

# Multifactor
Xk = sm.add_constant(df2[["MKT_EXCESS","SMB","HML","MOM"]])
mf = sm.OLS(df2["ASSET_EXCESS"], Xk).fit()

print("CAPM R2:", capm.rsquared)
print("MF   R2:", mf.rsquared)
print("\nCAPM params:\n", capm.params)
print("\nMF params:\n", mf.params)


# %%
def summarize_model(m):
    out = pd.DataFrame({
        "coef": m.params,
        "pval": m.pvalues
    })
    return out, m.rsquared

capm_table, capm_r2 = summarize_model(capm)
mf_table, mf_r2 = summarize_model(mf)

print("CAPM R2:", capm_r2)
display(capm_table)

print("MF R2:", mf_r2)
display(mf_table)


# %%

tickers = ["AAPL", "MSFT", "^GSPC"]  # activo(s) + mercado
data = yf.download(tickers, start="2018-01-01", progress=False)["Close"].dropna()

rets = to_returns(data, method="log")
asset = rets["AAPL"]
mkt = rets["^GSPC"]

# Proxy RF simple (mejor: usar T-bills o FRED; aquí usamos 0 para demo)
rf = 0.0

df = pd.DataFrame({
    "ASSET_EXCESS": asset - rf,
    "MKT_EXCESS": mkt - rf,
    "MSFT": rets["MSFT"] - rf
}).dropna()

X = sm.add_constant(df["MKT_EXCESS"])
capm_real = sm.OLS(df["ASSET_EXCESS"], X).fit()
print(capm_real.summary())

X = sm.add_constant(df[["MKT_EXCESS", "MSFT"]])
multifactor = sm.OLS(df["ASSET_EXCESS"], X).fit()
print(multifactor.summary())


# %% [markdown]
# ## Data real: Fama-French-Carhart

# %%
# !pip install fredapi

# %%
from fredapi import Fred
fred = Fred(api_key = key)

# %%
fred.search('risk free').head(3)

# %%
# tasa libre de riesgo: 3 meses
risk_free = fred.get_series('GS3M')
# risk_free.head(2)
risk_free = risk_free['2010-01-01':'2026-03-01']
rf = risk_free/100
rf.head(5)

# %%
rf = rf.resample('Q').mean()*3
rf.head(5)

# %%
plt.plot(risk_free)
plt.xlabel('Date')
plt.ylabel('%')
plt.title('3-month Treasury Constant Maturity Rate')

# %%
# GDP 
gdp = fred.get_series('GDP')
gdp = gdp['2010-01-01':'2026-03-01']
gdp.head(5)

# %%
gdp_growth = gdp.pct_change().dropna()
gdp_growth.tail()

# %%
plt.plot(gdp_growth)
plt.xlabel('Date')
plt.ylabel('GDP Growth Rate')
plt.title('GDP Growth, 2010-2026')

# %%
fred.search('potencial inflation')
inf = fred.get_series('CPIEALL')
inf = inf['2010-01-01':'2026-03-01']
inf.tail()

# %%
inf_quarterly = inf.resample('Q').mean()
inf_growth = inf_quarterly.pct_change().dropna()
inf_growth.tail()

# %%
plt.plot(inf_growth)
plt.xlabel('Date')
plt.ylabel('Inflation Growth Rate')
plt.title('Inflation Growth Rate, 2010-2026')

# %%
rics = ['JPM', 'V', 'MA', 'MS', 'GS',
'XOM','CVX', 'SPY']
start_date = '2014-01-01'
end_date = '2026-03-01'
data = yf.download(rics, start=start_date, interval='3mo', progress=False)["Close"].dropna()
rets = to_returns(data)
rets.reset_index(inplace=True)
print(rets.head(3))
print("----------------")
print("Num obs: ", rets.shape[0])
print("min date:", rets.index.min())
print("max date:", rets.index.max())

# %%
# tasa libre de riesgo: 3 meses
risk_free = fred.get_series('GS3M')
# risk_free.head(2)
risk_free = risk_free[start_date:end_date]
rf = pd.DataFrame(risk_free/100, columns=['rf'])
rf.reset_index(inplace=True)
rf.rename(columns={'index': 'Date'}, inplace=True)
rf['Date'] = pd.to_datetime(rf['Date'])
print(rf.head(5))
print("min date:", rf.Date.min())
print("max date:", rf.Date.max())


# %%
# GDP
gdp = fred.get_series('A191RP1Q027SBEA')
# gdp = fred.get_series('GDP')
gdp = gdp[start_date:end_date]
gdp = pd.DataFrame(gdp, columns=['gdp'])
gdp.reset_index(inplace=True)
gdp.rename(columns={'index': 'Date'}, inplace=True)
gdp['Date'] = pd.to_datetime(gdp['Date'])
print(gdp.tail(5))
print("min date:", gdp.Date.min())
print("max date:", gdp.Date.max())

# %%
# inflacion IPC
fred.search('potencial inflation')
inf = fred.get_series('CPIEALL')
inf = inf[start_date:end_date]
inf = pd.DataFrame(inf, columns=['ipc'])
inf['Date'] = pd.to_datetime(inf.index)
print(inf.tail(5))
print("min date:", inf.Date.min())
print("max date:", inf.Date.max())

# %%
inf_quarterly = inf.resample('Q').mean()
inf_growth = inf_quarterly.pct_change().dropna()
inf_growth.reset_index(inplace=True)
inf_growth.rename(columns={'index': 'Date'}, inplace=True)
inf_growth['Date'] = pd.to_datetime(inf_growth['Date'])
inf_growth.head()

# %%
# petroleo
oil_price = yf.download('USO', start=start_date, interval='3mo', progress=False)['Close']
oil_price = to_returns(oil_price)
oil_price.reset_index(inplace=True)
oil_price.rename(columns={'index': 'Date'}, inplace=True)
oil_price.head(3)

# %%
rets.info()

# %%
df_final = rets.merge( rf, on = 'Date', how='inner').merge(gdp, on = 'Date', how='inner').merge(inf, on = 'Date', how='inner').merge(oil_price.rename(columns={'Close': 'oil_price'}), on = 'Date', how='inner').dropna()
print("min date:", df_final.Date.min())
print("max date:", df_final.Date.max())

# %%
for i in rics:
  df_final["excess_rend " + str(i)] = df_final[i] - df_final['rf']

# %%
df_final.rename(columns={'excess_rend SPY': 'market excess'}, inplace=True)
df_final.columns

# %%
# Ajuste Chevron con factores macroeconómicos
y = df_final['excess_rend XOM']
variables = ['USO', 'gdp', 'excess_rend CVX'] #['market excess', 'gdp', 'ipc', 'oil_price']
X1 = sm.add_constant(df_final[variables])
model = sm.OLS(y, X1).fit()

model_table, model_r2 = summarize_model(model)

print("Modelo R2:", model_r2)
display(model_table)
print("Resume del modelo:")
print(model.summary())

# %%


# %%



