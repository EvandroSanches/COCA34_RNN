### Rede Neural capaz de prever preço e tendencia da ação COCA34 baseando-se nos registros dos ultimos 100 dias
Atributos previstos
- Preço de Abertura
- Preço de Alta
- Preço de Baixa
- Preço de Fechamento

### Periodo utilizado: Diário
### !!!Atenção: o modelo de séries temporais não é capaz de prever preços a mais que um periodo de tempo a frente. Em um caso onde é previsto dois periodos a frente é necessario utilizar a previsão anterior como entrada de dados, isso distorce os resultados.
### Foram utilizados dados conhecidos para análise de resultado. Exemplo: a previsão 1 foi gerada com os 100 ultimos dias de uma base de dados conhecida, a previsão 2 foi gerada com os 100 ultimos dias em relação ao dia 1 de uma base de dados conhecida e assim por diante, prevendo sempre somente 1 dia a frente e gerando um gráfico de previsões. 
## Treinamento

![Screenshot_1](https://github.com/EvandroSanches/COCA34_RNN/assets/102191806/c2d49bec-243b-40c8-a0ac-57b3293c88eb)


# Resultados de Testes

## Abertura
![Screenshot_2](https://github.com/EvandroSanches/COCA34_RNN/assets/102191806/bc162979-08ec-4139-9f92-b45462254b67)


## Alta
![Screenshot_3](https://github.com/EvandroSanches/COCA34_RNN/assets/102191806/0ad0c99a-6822-47ca-bd19-8b0194c99be6)

## Baixa
![Screenshot_4](https://github.com/EvandroSanches/COCA34_RNN/assets/102191806/f5803441-8d56-428e-aba2-d92e6b503602)

## Fechamento
![Screenshot_5](https://github.com/EvandroSanches/COCA34_RNN/assets/102191806/6e4e245e-0104-4c7e-bc23-e98a293e70b4)

## Conclusão: 
### Apesar de resultados relativamentes bons, a performance do modelo foi afetada pela quantidade de atributos previstos, tendo um resultado inferior a um caso em que é previsto apenas um atributo.

