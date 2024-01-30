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


#Variaveis de controle
batch_size = 4 # - Define batch de treinamento
epochs = 20 # - Define epocas de treinamento
timesteps = 100 # - Define a quantidade de dias anteriores que serão usados para treino e previsão
dias = 160 # - Define quantos dias a frente serão previstos

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
                dados[i, 0:4] = modelo.predict(x)


        previsores = np.asarray(previsores)


        return previsores, preco_real

#Estrutura da rede neural
def CriaRede():
    modelo = Sequential()

    modelo.add(LSTM(units=100, return_sequences=True, input_shape=(timesteps, 4)))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80))
    modelo.add(Dropout(0.4))

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

def Previsao(caminho, mode):
    #Carregando dados e modelo ja treinado
    previsores, preco_real = CarregaDados(caminho, mode)
    dados = pd.read_csv('COCA34.SA_Previsao.csv')

    dados = dados.loc[dados.index >= 100]
    dados['Date'] = pd.to_datetime(dados['Date'])

    modelo = load_model('Modelo.0.1')

    result = modelo.predict(previsores)

    #Desnormalizando dados
    normalizador = joblib.load('normalizador')
    result = normalizador.inverse_transform(result)
    preco_real = normalizador.inverse_transform(preco_real)

    colunas = ['Abertura', 'Alta', 'Baixa', 'Fechamento']
    result = pd.DataFrame(result, index=dados['Date'])
    preco_real = pd.DataFrame(preco_real, index=dados['Date'])

    for i in range(result.shape[1]):
        #Relatório de previsão
        plt.plot(result[i])
        plt.plot(preco_real[i])
        plt.title('Relação Previsão e Preço Real')
        plt.legend(['Previsão '+colunas[i], 'Preço Real '+colunas[i]])
        plt.xlabel('Data')
        plt.ylabel('R$')
        plt.show()



#Previsao(caminho do arquivo de dados no formato csv, modo de carregamento de dados)
#Modo de carregamento de dados
# - Treinamento : modelo ira carregar dados normaliza-los e estruturar dividindo em variaveis 'previsores' e 'preco_real' para comparação e analise
# - Previsao : modelo ira carregar dados normaliza-los, estruturar com divisão de variaveis 'previsores' e 'preco_real', porém,
# a variavel 'previsores' irá se autoalimentar com as previsões, permitindo prever o preço além dos dados ja conhecidos
# Exemplo: se possuir dados dos ultimos 100 dias no modo treinamento só sera possivel prever o preço do dia 101. No modo
# previsão é possivel prever além
# o modo Treinamento performou muito bem, porém, o modo previsão performou muito mal, a previsão recursiva estabiliza os dados, sendo assim muito dificil
# prever grandes avarias


Previsao('COCA34.SA_Previsao.csv','treinamento')
