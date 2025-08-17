## Azure AI Evaluation 

### Agent variants

| Agent name | Agent ID | Evaluation results |
|:-----------|:---------|:-------------------|
| test-agent-no-delete-01 | [asst_7ZijCsebAPQ7ggOkqJE1uH4G](https://ai.azure.com/playground/agents?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&assistantId=asst_7ZijCsebAPQ7ggOkqJE1uH4G "") | [Click here](https://ai.azure.com/resource/build/evaluation/996f2633-e741-414e-8151-82782f60d6fc?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&tid=72f988bf-86f1-41af-91ab-2d7cd011db47 "") |
| test-agent-no-delete-02 | [asst_q1l6UXTwGOa9smSOjRGROCEg](https://ai.azure.com/playground/agents?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&assistantId=asst_q1l6UXTwGOa9smSOjRGROCEg "") | [Click here](https://ai.azure.com/resource/build/evaluation/1cd3f50b-dc52-429f-8928-eb7c86a1bcbb?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&tid=72f988bf-86f1-41af-91ab-2d7cd011db47 "") |

### Compare evaluation scores between variants

#### Operational metrics

| Evaluation score        | test-agent-no-delete-01                                                           | test-agent-no-delete-02                                                                                                                                                              |
|:------------------------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Client run duration [s] | ![Baseline: 5.17](https://img.shields.io/badge/Baseline-5.17-ffffff "")           | ![Too few samples: 6.3 (+1.13)](https://img.shields.io/badge/Too%20few%20samples-6.3%20%28%2B1.13%29-f0e543 "Insufficient observations to determine statistical significance")       |
| Server run duration [s] | ![Baseline: 3](https://img.shields.io/badge/Baseline-3-ffffff "")                 | ![Too few samples: 4 (+1)](https://img.shields.io/badge/Too%20few%20samples-4%20%28%2B1%29-f0e543 "Insufficient observations to determine statistical significance")                 |
| Completion tokens       | ![Baseline: 225](https://img.shields.io/badge/Baseline-225-ffffff "")             | ![Too few samples: 320 (+95)](https://img.shields.io/badge/Too%20few%20samples-320%20%28%2B95%29-f0e543 "Insufficient observations to determine statistical significance")           |
| Prompt tokens           | ![Baseline: 2.19e+03](https://img.shields.io/badge/Baseline-2.19e%2B03-ffffff "") | ![Too few samples: 2.19e+03 (+0)](https://img.shields.io/badge/Too%20few%20samples-2.19e%2B03%20%28%2B0%29-f0e543 "Insufficient observations to determine statistical significance") |

#### AI quality (AI assisted)

| Evaluation score                | test-agent-no-delete-01                                                       | test-agent-no-delete-02                                                                                                                                                                  |
|:--------------------------------|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tool Call Accuracy passing rate | ![Baseline: 100.0%](https://img.shields.io/badge/Baseline-100.0%25-ffffff "") | ![Too few samples: 100.0% (+0.0%)](https://img.shields.io/badge/Too%20few%20samples-100.0%25%20%28%2B0.0%25%29-f0e543 "Insufficient observations to determine statistical significance") |
| Fluency passing rate            | ![Baseline: 100.0%](https://img.shields.io/badge/Baseline-100.0%25-ffffff "") | ![Too few samples: 100.0% (+0.0%)](https://img.shields.io/badge/Too%20few%20samples-100.0%25%20%28%2B0.0%25%29-f0e543 "Insufficient observations to determine statistical significance") |
| Groundedness passing rate       | ![Baseline: 100.0%](https://img.shields.io/badge/Baseline-100.0%25-ffffff "") | ![Too few samples: 100.0% (+0.0%)](https://img.shields.io/badge/Too%20few%20samples-100.0%25%20%28%2B0.0%25%29-f0e543 "Insufficient observations to determine statistical significance") |

### References

- See [evaluator-scores.yaml](https://github.com/microsoft/ai-agent-evals/blob/main/analysis/evaluator-scores.yaml) for the full list of evaluators supported and the definitions of the scores
- For in-depth details on evaluators, please see the [Agent Evaluation SDK section in the Azure AI documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/agent-evaluate-sdk)
## Azure AI Evaluation 

### Agent variants

| Agent name | Agent ID | Evaluation results |
|:-----------|:---------|:-------------------|
| test-agent-no-delete-01 | [asst_7ZijCsebAPQ7ggOkqJE1uH4G](https://ai.azure.com/playground/agents?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&assistantId=asst_7ZijCsebAPQ7ggOkqJE1uH4G "") | [Click here](https://ai.azure.com/resource/build/evaluation/5a09266c-660a-46a7-9c86-481461209bc0?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&tid=72f988bf-86f1-41af-91ab-2d7cd011db47 "") |
| test-agent-no-delete-02 | [asst_q1l6UXTwGOa9smSOjRGROCEg](https://ai.azure.com/playground/agents?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&assistantId=asst_q1l6UXTwGOa9smSOjRGROCEg "") | [Click here](https://ai.azure.com/resource/build/evaluation/b783a425-2a16-492c-bd96-13c39b32c57b?wsid=/subscriptions/040c3674-68d4-431e-b456-a8542c77b8d1/resourceGroups/rg-aprilk-gp-eval-01/providers/Microsoft.CognitiveServices/accounts/aoai-qyrftgva6obps/projects/proj-qyrftgva6obps&tid=72f988bf-86f1-41af-91ab-2d7cd011db47 "") |

### Compare evaluation scores between variants

#### Operational metrics

| Evaluation score        | test-agent-no-delete-01                                                         | test-agent-no-delete-02                                                                                                                                                               |
|:------------------------|:--------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Client run duration [s] | ![Baseline: 8.16](https://img.shields.io/badge/Baseline-8.16-ffffff "")         | ![Too few samples: 7.59 (-0.572)](https://img.shields.io/badge/Too%20few%20samples-7.59%20%28--0.572%29-f0e543 "Insufficient observations to determine statistical significance")     |
| Server run duration [s] | ![Baseline: 7](https://img.shields.io/badge/Baseline-7-ffffff "")               | ![Too few samples: 6 (-1)](https://img.shields.io/badge/Too%20few%20samples-6%20%28--1%29-f0e543 "Insufficient observations to determine statistical significance")                   |
| Completion tokens       | ![Baseline: 282](https://img.shields.io/badge/Baseline-282-ffffff "")           | ![Too few samples: 243 (-39)](https://img.shields.io/badge/Too%20few%20samples-243%20%28--39%29-f0e543 "Insufficient observations to determine statistical significance")             |
| Prompt tokens           | ![Baseline: 2.2e+03](https://img.shields.io/badge/Baseline-2.2e%2B03-ffffff "") | ![Too few samples: 2.19e+03 (-11)](https://img.shields.io/badge/Too%20few%20samples-2.19e%2B03%20%28--11%29-f0e543 "Insufficient observations to determine statistical significance") |

#### AI quality (AI assisted)

| Evaluation score                | test-agent-no-delete-01                                                       | test-agent-no-delete-02                                                                                                                                                                  |
|:--------------------------------|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tool Call Accuracy passing rate | ![Baseline: 100.0%](https://img.shields.io/badge/Baseline-100.0%25-ffffff "") | ![Too few samples: 100.0% (+0.0%)](https://img.shields.io/badge/Too%20few%20samples-100.0%25%20%28%2B0.0%25%29-f0e543 "Insufficient observations to determine statistical significance") |
| Fluency passing rate            | ![Baseline: 100.0%](https://img.shields.io/badge/Baseline-100.0%25-ffffff "") | ![Too few samples: 100.0% (+0.0%)](https://img.shields.io/badge/Too%20few%20samples-100.0%25%20%28%2B0.0%25%29-f0e543 "Insufficient observations to determine statistical significance") |
| Groundedness passing rate       | ![Baseline: 100.0%](https://img.shields.io/badge/Baseline-100.0%25-ffffff "") | ![Too few samples: 100.0% (+0.0%)](https://img.shields.io/badge/Too%20few%20samples-100.0%25%20%28%2B0.0%25%29-f0e543 "Insufficient observations to determine statistical significance") |

### References

- See [evaluator-scores.yaml](https://github.com/microsoft/ai-agent-evals/blob/main/analysis/evaluator-scores.yaml) for the full list of evaluators supported and the definitions of the scores
- For in-depth details on evaluators, please see the [Agent Evaluation SDK section in the Azure AI documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/agent-evaluate-sdk)
