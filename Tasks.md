## Tasks:
- 1. agora acho que ia tentar melhorar o modelo de previsão  
- 2. os valores de ocupação no ficheiro JSON nao estão corretos, por exemplo em 
```JSON
{
        "t": 85,
        "occ_P023": 0.011111111111111112,
        "price_P023": 1.5,
        "evutil_P023": 0.0
      },
      {
        "t": 90,
        "occ_P023": 0.011111111111111112,
        "price_P023": 1.5,
        "evutil_P023": 0.0
      },
      {
        "t": 95,
        "occ_P023": 0.011111111111111112,
        "price_P023": 1.5,
        "evutil_P023": 0.0
      },
      {
        "t": 100,
        "occ_P023": 0.011111111111111112,
        "price_P023": 1.5,
        "evutil_P023": 0.0
      },
      {
        "t": 105,
        "occ_P023": 0.005555555555555556,
        "price_P023": 1.5,
        "evutil_P023": 0.0
      },
      {
        "t": 110,
        "occ_P023": 0.005555555555555556,
        "price_P023": 1.5,
        "evutil_P023": 0.0
      },
      {
        "t": 115,
        "occ_P023": 0.005555555555555556,
        "price_P023": 1.5,
        "evutil_P023": 0.0
      },
```
- 3. o que significa "evutil_P023": 0.0 -> é o carregamento Eletrico (stand-by) 
- 4. o que falta em si corrigir entre estes tópicos: 
	- 4.1. Perfil diário de ocupação
		- 4.1.1. Recolhe-se a ocupação segunda a sexta.
		- 4.1.2. Dados FOS (provavelmente “Full Occupancy Schedule”).
		- 4.1.3. Perfil de utilização para todos os dias durante 6 meses.
		- 4.1.4. Visualização temporal (curva de ocupação ao longo do dia → “pernoita” visível).
	- 4.2. Modelação descritiva 
		- 4.2.1. Criar modelos descritivos baseados em distribuições estatísticas dos padrões observados.
		- 4.2.2. Dividir os dados em intervalos de 1/2 hora.
		- 4.2.3. Estimar distribuições estatísticas (ex. Normal N(μ, σ²)) para cada intervalo.
		- 4.2.4. Possível usar curve fitting ou jittering para aproximar padrões.
		- 4.2.5. Objetivo: gerar ocupações sintéticas mas realistas para simulação.
	- 4.3. Digital Twin (done)
		- 4.3.1. É um modelo descritivo que simula a realidade do parque.
		- 4.3.2. Permite testar políticas de preços e modificações (ex: número de lugares).
		- 4.3.3. Baseado num modelo de ocupação por chegada (entrada/saída de carros).
		- 4.3.4. Inputs: 
			- 4.3.4.1. Dados reais de ocupação, localização, tráfego, número de lugares.
			- 4.3.4.2. Dados do histórico de entradas e saídas.
			- 4.3.4.3. Permite prever comportamento futuro e avaliar cenários.
		- 4.3.5. Outputs:
			- 4.3.5.1. Modelo de tráfego nas redondezas do parque.
			- 4.3.5.2. Ocupação ao longo do tempo (e.g., 2 em 2 horas).
			- 4.3.5.3. Análise de capacidade e transição (entradas/saídas).
	- 4.4. Simulação e calibração
		- 4.4.1. Criar datasets com transições de estado (entrar/sair do parque). 
		- 4.4.2. Exemplo: 20 carros por hora = tráfego moderado; 30 carros → pico. 
		- 4.4.3. Avaliar fluxo urbano (percentagem de carros que entram no parque vs passam ao lado).
		- 4.4.4. Considerar probabilidades condicionais:
			- 4.4.4.1. P(carro entra no parque).
			- 4.4.4.2. P(carro sai do parque | trânsito confortável).
			- 4.4.4.3. Base para o modelo de decisão.
	- 4.5. Generalização
		- 4.5.1. Fazer o modelo para um parque piloto. (done)
		- 4.5.2. Depois, generalizar a metodologia para outros parques.
		- 4.5.3. Avaliar escalabilidade e capacidade de previsão.

    ----

### O que falta:
4.4.3. Avaliar fluxo urbano

- Avaliar a percentagem de carros que entram no parque vs passam ao lado (Taxa de - Rejeição/Perda de Clientes devido ao preço ou trânsito).

4.4.4. Considerar probabilidades condicionais:

- 4.4.4.2. P(carro sai do parque | trânsito confortável). (Atualmente o modelo assume que a saída é fixa pelo histórico, não considera se está difícil sair devido ao trânsito).

4.5. Generalização

- 4.5.2. Depois, generalizar a metodologia para outros parques (ex: testar o código com o parque P064).

- 4.5.3. Avaliar escalabilidade e capacidade de previsão (verificar se o modelo mantém a precisão noutros cenários).


---

Testar no yaml (varios modelos de preço):
Conservador: target: 0.7, k: 1.0, p_min: 1.0, p_max: 2.0

Agressivo: target: 0.9, k: 3.0, p_min: 1.0, p_max: 3.0

Lento: interval: 15 (preço ajusta-se mais devagar)