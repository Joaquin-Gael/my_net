from models.net import NeuronalNet
import polars as pl
import numpy as np

def minmax_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val

def minmax_denormalize(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val

df_product_price = pl.read_csv('data/product_prices.csv')
df_product_price = df_product_price.with_columns(
    pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')\
        .cast(pl.Int64)
)

nn = NeuronalNet()

dates_matriz = df_product_price.select('date').to_numpy().reshape(-1, 1)
prices_matriz = df_product_price.select('price').to_numpy().reshape(-1, 1)

dates_matriz_norm, dates_min, dates_max = minmax_normalize(dates_matriz)
prices_matriz_norm, prices_min, prices_max = minmax_normalize(prices_matriz)

#print(dates_matriz_norm.shape)
#print(dates_matriz_norm)
#print(prices_matriz_norm.shape)
#print(prices_matriz_norm)

train_x = dates_matriz[1:int(len(dates_matriz_norm) * 0.6)]
train_y = prices_matriz[1:int(len(prices_matriz_norm) * 0.6)]

#print(train_x.shape)
#print(train_x)
#print(train_y.shape)
#print(train_y)

nn.add_layer(1, 6, activation='relu')
nn.add_layer(6, 12, activation='relu')
nn.add_layer(12, 6, activation='relu')
nn.add_layer(6, 1, activation='relu')

nn.set_lr(0.00000001)

nn.fit(train_x, train_y, batch_size=int(len(train_x) / 3), epochs=1000)

test_x = dates_matriz[int(len(dates_matriz_norm) * 0.9):]
test_y = prices_matriz[int(len(prices_matriz_norm) * 0.9):]

#print(test_x.shape)

print('des normalizados: ',minmax_denormalize(nn.predict(test_x), prices_min, prices_max))
#print(1 / (1 + np.exp(np.clip(-2.61897518, -5, 5))))
#print('normalizados: ',nn.predict(test_x))

#print(prices_matriz_norm)
#print(dates_matriz_norm)
#print(minmax_denormalize(prices_matriz_norm, prices_min, prices_max))

#print('original:\n', test_y)