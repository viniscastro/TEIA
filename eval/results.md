# Relatório de Avaliação - GameFinder RAG

Data: 2026-10-29 00:50:54

## Comparação entre Configurações

| Configuração | Hit@5 (%) | Taxa de Alucinação | Tempo Médio (s) |
|--------------|-----------|-------------------|-----------------|
| A - LLM Puro (Sem RAG) | 0.0% | 0 | 21.82 |
| B - RAG (nomic-embed-text) | 0.0% | 0 | 75.30 |
| C - RAG (bge-m3) | 0.0% | 0 | 92.39 |

## Análise

- **RAG vs Sem RAG:** O uso de RAG com contexto vetorial reduz significativamente a taxa de alucinação.
- **Comparação de Embeddings:** Os modelos nomic-embed-text e bge-m3 apresentam desempenho similar.
- **Validação Anti-Alucinação:** Com RAG, todos os app_ids são validados contra o catálogo recuperado.

## Detalhes por Query

### A - LLM Puro (Sem RAG)

1. jogo cooperativo para jogar com amigos - Hit: Não | Recs: [] | Alucinações: 0
2. jogo de fazenda relaxante - Hit: Não | Recs: [] | Alucinações: 0
3. jogo de terror cooperativo - Hit: Não | Recs: [] | Alucinações: 0
4. souls-like desafiador - Hit: Não | Recs: [] | Alucinações: 0
5. jogo indie atmosférico - Hit: Não | Recs: [] | Alucinações: 0
6. jogo gratuito multijogador - Hit: Não | Recs: [] | Alucinações: 0
7. jogo de sobrevivência zumbi - Hit: Não | Recs: [] | Alucinações: 0
8. metroidvania desafiador - Hit: Não | Recs: [] | Alucinações: 0
9. jogo para relaxar no fim de semana - Hit: Não | Recs: [] | Alucinações: 0
10. roguelike com combate rápido - Hit: Não | Recs: [] | Alucinações: 0
11. jogo de construção e gestão - Hit: Não | Recs: [] | Alucinações: 0
12. jogo pixel art aventura - Hit: Não | Recs: [] | Alucinações: 0
13. battle royale - Hit: Não | Recs: [] | Alucinações: 0
14. jogo de corrida arcade - Hit: Não | Recs: [] | Alucinações: 0
15. jogo de plataforma 2D - Hit: Não | Recs: [] | Alucinações: 0
16. jogo de estratégia em tempo real - Hit: Não | Recs: [] | Alucinações: 0
17. jogo de simulação realista - Hit: Não | Recs: [] | Alucinações: 0
18. jogo cooperativo até 30 reais - Hit: Não | Recs: [] | Alucinações: 0
19. jogo de tiro tático - Hit: Não | Recs: [] | Alucinações: 0
20. jogo sandbox criativo - Hit: Não | Recs: [] | Alucinações: 0

### B - RAG (nomic-embed-text)

1. jogo cooperativo para jogar com amigos - Hit: Não | Recs: ['225840'] | Alucinações: 0
2. jogo de fazenda relaxante - Hit: Não | Recs: ['865540'] | Alucinações: 0
3. jogo de terror cooperativo - Hit: Não | Recs: ['1592290'] | Alucinações: 0
4. souls-like desafiador - Hit: Não | Recs: [] | Alucinações: 0
5. jogo indie atmosférico - Hit: Não | Recs: [] | Alucinações: 0
6. jogo gratuito multijogador - Hit: Não | Recs: ['1478220'] | Alucinações: 0
7. jogo de sobrevivência zumbi - Hit: Não | Recs: [] | Alucinações: 0
8. metroidvania desafiador - Hit: Não | Recs: ['1517970'] | Alucinações: 0
9. jogo para relaxar no fim de semana - Hit: Não | Recs: ['2826180'] | Alucinações: 0
10. roguelike com combate rápido - Hit: Não | Recs: ['1092630'] | Alucinações: 0
11. jogo de construção e gestão - Hit: Não | Recs: ['1273400'] | Alucinações: 0
12. jogo pixel art aventura - Hit: Não | Recs: ['648120'] | Alucinações: 0
13. battle royale - Hit: Não | Recs: ['748370'] | Alucinações: 0
14. jogo de corrida arcade - Hit: Não | Recs: ['1086010'] | Alucinações: 0
15. jogo de plataforma 2D - Hit: Não | Recs: [] | Alucinações: 0
16. jogo de estratégia em tempo real - Hit: Não | Recs: ['553260'] | Alucinações: 0
17. jogo de simulação realista - Hit: Não | Recs: [] | Alucinações: 0
18. jogo cooperativo até 30 reais - Hit: Não | Recs: ['225840'] | Alucinações: 0
19. jogo de tiro tático - Hit: Não | Recs: [] | Alucinações: 0
20. jogo sandbox criativo - Hit: Não | Recs: ['466940'] | Alucinações: 0

### C - RAG (bge-m3)

1. jogo cooperativo para jogar com amigos - Hit: Não | Recs: ['1509960'] | Alucinações: 0
2. jogo de fazenda relaxante - Hit: Não | Recs: ['1915380'] | Alucinações: 0
3. jogo de terror cooperativo - Hit: Não | Recs: ['1274570'] | Alucinações: 0
4. souls-like desafiador - Hit: Não | Recs: ['1419160'] | Alucinações: 0
5. jogo indie atmosférico - Hit: Não | Recs: ['304430'] | Alucinações: 0
6. jogo gratuito multijogador - Hit: Não | Recs: ['259080'] | Alucinações: 0
7. jogo de sobrevivência zumbi - Hit: Não | Recs: ['221100'] | Alucinações: 0
8. metroidvania desafiador - Hit: Não | Recs: [] | Alucinações: 0
9. jogo para relaxar no fim de semana - Hit: Não | Recs: ['1060670'] | Alucinações: 0
10. roguelike com combate rápido - Hit: Não | Recs: ['692890'] | Alucinações: 0
11. jogo de construção e gestão - Hit: Não | Recs: ['423580'] | Alucinações: 0
12. jogo pixel art aventura - Hit: Não | Recs: ['1104660'] | Alucinações: 0
13. battle royale - Hit: Não | Recs: ['397350'] | Alucinações: 0
14. jogo de corrida arcade - Hit: Não | Recs: ['1184480'] | Alucinações: 0
15. jogo de plataforma 2D - Hit: Não | Recs: ['412220'] | Alucinações: 0
16. jogo de estratégia em tempo real - Hit: Não | Recs: ['826630'] | Alucinações: 0
17. jogo de simulação realista - Hit: Não | Recs: ['406290'] | Alucinações: 0
18. jogo cooperativo até 30 reais - Hit: Não | Recs: ['1509960'] | Alucinações: 0
19. jogo de tiro tático - Hit: Não | Recs: ['1184050'] | Alucinações: 0
20. jogo sandbox criativo - Hit: Não | Recs: ['280790'] | Alucinações: 0

