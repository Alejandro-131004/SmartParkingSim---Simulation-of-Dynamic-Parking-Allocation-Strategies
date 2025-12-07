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
		- Recolhe-se a ocupação segunda a sexta.
		- Dados FOS (provavelmente “Full Occupancy Schedule”).
		- Perfil de utilização para todos os dias durante 6 meses.
		- Visualização temporal (curva de ocupação ao longo do dia → “pernoita” visível).
	- 4.2. Modelação descritiva
		- Criar modelos descritivos baseados em distribuições estatísticas dos padrões observados.
		- Dividir os dados em intervalos de 1/2 hora.
		- Estimar distribuições estatísticas (ex. Normal N(μ, σ²)) para cada intervalo.
		- Possível usar curve fitting ou jittering para aproximar padrões.
		- Objetivo: gerar ocupações sintéticas mas realistas para simulação.
	- 4.3. Digital Twin
		- É um modelo descritivo que simula a realidade do parque.
		- Permite testar políticas de preços e modificações (ex: número de lugares).
		- Baseado num modelo de ocupação por chegada (entrada/saída de carros).
		- Inputs:
			- Dados reais de ocupação, localização, tráfego, número de lugares.
			- Dados do histórico de entradas e saídas.
			- Permite prever comportamento futuro e avaliar cenários.
		- Outputs:
			- Modelo de tráfego nas redondezas do parque.
			- Ocupação ao longo do tempo (e.g., 2 em 2 horas).
			- Análise de capacidade e transição (entradas/saídas).
	- 4.4. Simulação e calibração
		- Criar datasets com transições de estado (entrar/sair do parque).
		- Exemplo: 20 carros por hora = tráfego moderado; 30 carros → pico.
		- Avaliar fluxo urbano (percentagem de carros que entram no parque vs passam ao lado).
		- Considerar probabilidades condicionais:
			- P(carro entra no parque).
			- P(carro sai do parque | trânsito confortável).
			- Base para o modelo de decisão.
	- 4.5. Generalização
		- Fazer o modelo para um parque piloto.
		- Depois, generalizar a metodologia para outros parques.
		- Avaliar escalabilidade e capacidade de previsão.