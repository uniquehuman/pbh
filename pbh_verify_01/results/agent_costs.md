# agent_costs.md — фактические расходы субагентов (из уведомлений среды)

| Роль | Архитектура | Начало (UTC) | Длительность, с | Токены субагента | Вызовов инструментов |
|---|---|---|---|---|---|
| Physics derivation | B | 22:31 | 1248 | 230181 | 40 |
| Numerical implementation | B | 23:00 | 1500 | 247166 | 49 |
| Solo agent (implement + self-review) | A | 22:52 | 2429 | 278182 | 69 |
| Adversarial audit (seeded copy) | B | 23:19 | 856 | 189057 | 39 |
| Independent verification (phase 1) | B | 22:31 | 4726 | 337897 | 70 |
| Numerical implementation, round 2 (v0.2) | B | 23:54 | 701 | 324430 (cumulative for the agent, incl. round 1) | 29 |
| Numerical implementation, round 3 (v0.3) | B | 00:19 | 186 | 344008 (cumulative) | 10 |
| Independent verification, phase 2 (matrix v2) | B | 23:54 | 7061 | 482447 (cumulative for the agent) | 57 |
| Numerical implementation, round 4 (v0.4) | B | 01:57 | 164 | 364151 (cumulative) | 10 |
| Numerical implementation, round 5 (v0.5) | B | 02:12 | 168 | 378266 (cumulative) | 10 |
