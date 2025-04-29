## Azure AI Evaluation (all-scores)

### Agent variants

| Agent name | Agent ID | Evaluation results |
|:-----------|:---------|:-------------------|
| agent_version_1 | [agent.v1](https://ai-url/agent.v1 "") | [Click here](test_url_1 "") |
| agent_version_2 | [agent.v2](https://ai-url/agent.v2 "") | [Click here](test_url_1 "") |

### Compare evaluation scores between variants

#### AI quality (AI assisted)

| Evaluation score   | agent.v1                                                                | agent.v2                                                                                                                                                                        |
|:-------------------|:------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Relevance          | ![Baseline: 4.33](https://img.shields.io/badge/Baseline-4.33-ffffff "") | ![Too few samples: 4.00 (-0.33)](https://img.shields.io/badge/Too%20few%20samples-4.00%20%28--0.33%29-f0e543 "Insufficient observations to determine statistical significance") |
| Fluency            | ![Baseline: 0.85](https://img.shields.io/badge/Baseline-0.85-ffffff "") | ![Too few samples: 0.62 (-0.23)](https://img.shields.io/badge/Too%20few%20samples-0.62%20%28--0.23%29-f0e543 "Insufficient observations to determine statistical significance") |

### References

- See [evaluator-scores.yaml](https://github.com/microsoft/ai-agent-evals/blob/main/analysis/evaluator-scores.yaml) for the full list of evaluators supported and the definitions of the scores
- For in-depth details on evaluators, please see the [Agent Evaluation SDK section in the Azure AI documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/agent-evaluate-sdk)
