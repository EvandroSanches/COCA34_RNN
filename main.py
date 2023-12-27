from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from statistics import mean, stdev
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Rede Neurais Recorrentes - Previsão da ação COCA34
# Previsão com multiplas entradas e multiplas saidas

#Conclusão, em comparação aos outros, o modelo não se saiu muito bem tendo multiplas saidas de previsão

#Variaveis de controle
batch_size = 5 # - Define batch de treinamento
epochs = 100 # - Define epocas de treinamento
timesteps = 90 # - Define a quantidade de dias anteriores que serão usados para treino e previsão
dias = 15 # - Define quantos dias a frente serão previstos

def CarregaDados(caminho, mode):
    #Carregando e tratando base de dados
    dados = pd.read_csv(caminho)
    dados = dados.dropna()
    dados = dados.drop(['Date'], axis=1)
    dados = dados.drop(['Volume'], axis=1)
    dados = dados.drop(['Adj Close'], axis=1)

    #Normalizando dados
    normalizador = MinMaxScaler()
    dados = normalizador.fit_transform(dados)
    joblib.dump(normalizador, 'normalizador')

    #Definindo estrutura de dados de treinamento
    if mode.lower() == 'treinamento':
        previsores = []
        preco_real = []
        for i in range(timesteps, dados.shape[0]):
            previsores.append(dados[i - timesteps : i, 0:4])
            preco_real.append(dados[i , 0:4])

        previsores = np.asarray(previsores)
        preco_real = np.asarray(preco_real)

        return previsores, preco_real

    #Estruturando dados de Previsão com auto-alimentação
    #O modelo ira carregar os dados de previsão com as próprias previsões
    #Exemplo: Prever valor dos proximos 10 dias, modelo irá receber dados de dias anteriores irá prever o dia 01
    #na previsão do dia 02 irá usar a previsão do dia 01 como dado de valores anteriores
    elif mode.lower() == 'previsao':

        try:
            modelo = load_model('Modelo.0.1')
        except:
            print('Ainda não existe um modelo treinado!!')

        previsores = []
        preco_real = []

        for i in range(timesteps, dados.shape[0]):
            preco_real.append(dados[i , 0:4])

        preco_real = np.asarray(preco_real)

        for i in range(timesteps, dias+timesteps):
            previsores.append(dados[i - timesteps : i, 0:4])
            x = np.asarray(previsores[i - timesteps])
            x = np.expand_dims(x, axis=0)

            if i == dados.shape[0]:
                result = modelo.predict(x)
                dados = np.vstack((dados, result))
            else:
                dados[i] = modelo.predict(x)


        previsores = np.asarray(previsores)


        return previsores, preco_real

#Estrutura da rede neural
def CriaRede():
    modelo = Sequential()

    modelo.add(LSTM(units=120, return_sequences=True, input_shape=(timesteps, 4)))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=120, return_sequences=True))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=120, return_sequences=True))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=120, return_sequences=True))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=120))
    modelo.add(Dropout(0.3))

    modelo.add(Dense(units=4, activation='linear'))

    modelo.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return modelo


def Treinamento():
    #Carregando e tratando dados de treinamento
    previsores, preco_real = CarregaDados('COCA34.SA.csv','treinamento')
    previsores_teste, preco_real_teste = CarregaDados('COCA34.SA_Previsao.csv', 'Treinamento')

    modelo = CriaRede()

    #Definindo Callbacks
    ers = EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta= 1e-10 )
    rlp = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
    mc = ModelCheckpoint(filepath='Modelo.0.1', save_best_only=True, verbose=1 )

    resultado = modelo.fit(previsores, preco_real, batch_size=batch_size, epochs=epochs, validation_data=(previsores_teste,preco_real_teste), callbacks=[ers, rlp, mc])

    media = mean(resultado.history['val_mae'])
    desvio = stdev(resultado.history['val_mae'])

    #Relatório de Treinamento
    plt.plot(resultado.history['mae'])
    plt.plot(resultado.history['val_mae'])
    plt.title('Relação de Perda em R$\nMédia:'+str(media)+'\nDesvio Padrão:'+str(desvio))
    plt.xlabel('Épocas')
    plt.ylabel('R$')
    plt.legend(('Treinamento', 'Teste'))
    plt.show()

def Previsao(caminho):
    #Carregando dados e modelo ja treinado
    previsores, preco_real = CarregaDados(caminho, 'previsao')

    modelo = load_model('Modelo.0.1')

    result = modelo.predict(previsores)

    #Desnormalizando dados
    normalizador = joblib.load('normalizador')
    result = normalizador.inverse_transform(result)
    preco_real = normalizador.inverse_transform(preco_real)

    #Relatório de previsão
    plt.plot(result)
    plt.plot(preco_real)
    plt.title('Relação Previsão e Preço Real')
    plt.legend(['Previsão Abertura','Previsão Alta', 'Previsão Baixa', 'Previsão Fechamento', 'Preço Real Abertura', 'Preço Real Alta', 'Preço Real Baixa', 'Preço Real Fechamento'])
    plt.xlabel('Dias')
    plt.ylabel('R$')
    plt.show()

Treinamento()
Previsao('COCA34.SA_Previsao.csv')
