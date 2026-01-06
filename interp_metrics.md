# Relatório de Validação: Previsão de Ocupação (Baseline)

Este documento detalha as métricas de desempenho obtidas na validação do modelo estatístico de previsão de ocupação (`OccupancyForecaster`). O modelo foi testado num conjunto de dados real (Junho 2022) comparando a previsão com a ocupação observada.

## Resumo dos Resultados

| Métrica | Valor Obtido | Unidade |
| :--- | :--- | :--- |
| **MAE** | `0.0771` | Rácio (0-1) |
| **RMSE** | `0.1067` | Rácio (0-1) |
| **R²** | `0.5609` | Adimensional |

---

## 1. Análise do MAE (Mean Absolute Error)
**Valor:** `0.0771` (7.71%)

O **MAE** representa o erro médio absoluto entre a previsão e a realidade. Tratando-se de um parque com capacidade para **180 lugares**, este valor traduz-se no seguinte erro prático:

$$\text{Erro Médio} = 0.0771 \times 180 \approx \mathbf{13.9 \text{ carros}}$$

**Interpretação:**
Em condições normais de operação, o modelo erra, em média, por cerca de **14 carros**. Este é um valor aceitável para um *baseline* estatístico, indicando que o perfil diário médio (ciclos manhã/tarde) foi bem aprendido.

---

## 2. Análise do RMSE (Root Mean Squared Error)
**Valor:** `0.1067` (10.67%)

O **RMSE** penaliza erros de grande magnitude de forma quadrática. Convertendo para a capacidade do parque:

$$\text{Erro Penalizado} = 0.1067 \times 180 \approx \mathbf{19.2 \text{ carros}}$$

**Crítica (RMSE vs MAE):**
O RMSE é cerca de **40% superior** ao MAE (`0.107` vs `0.077`).
Esta discrepância indica que **a distribuição dos erros não é uniforme**. O modelo sofre de **erros pontuais severos**.
* Observando o gráfico de validação, existem dias em que a ocupação real cai para perto de zero (anomalias, feriados ou falhas de sensor), mas o modelo continua a prever uma ocupação normal.
* Nesses momentos, o erro dispara (ex: erro de 100 carros), o que "insufla" o RMSE.

---

## 3. Análise do R² (Coefficient of Determination)
**Valor:** `0.561`

O modelo explica aproximadamente **56% da variância** dos dados de teste.

**Veredito:**
* O modelo captura com sucesso a **sazonalidade diária** (a "forma" das ondas de ocupação repete-se corretamente todos os dias).
* No entanto, falha em capturar a **variância específica de cada dia** (eventos atípicos, dias de chuva intensa, feriados móveis). Como o modelo é puramente estatístico (baseado em médias históricas), ele não tem "consciência" de eventos externos, o que limita o R² a este patamar mediano.

## Conclusão Global

O `OccupancyForecaster` é validado como um modelo **robusto para dias típicos**, com um erro médio inferior a 8%. No entanto, a análise do RMSE revela a sua fragilidade perante anomalias e eventos não-cíclicos, onde o erro tende a agravar-se significativamente. Para o propósito de *Digital Twin* e simulação de preços, serve como uma base de referência estável.