# Azure AI Evaluation GitHub Action

This GitHub Action enables offline evaluation of Azure AI agents within your CI/CD pipelines. It is designed to streamline the evaluation process, allowing you to assess agent performance and make informed decisions before deploying to production.

Offline evaluation involves testing AI agents using test datasets to measure their performance on various quality and safety metrics such as fluency, coherence and content safety. After setting up an [Azure AI Agent Service](hhttps://learn.microsoft.com/en-us/azure/ai-services/agents/), offline pre-production evaluation is crucial for AI application validation during integration testing, allowing developers to identify potential issues and make improvements before releasing an update to your Azure AI agent.

## Features

- **Automated Evaluation:** Integrate offline evaluation into your CI/CD workflows to automate the pre-production assessment of Azure AI agents.
- **Built-in Evaluators:** Leverage existing evaluators provided by the [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/evaluate-sdk). The following evaluators are supported: TODO
- **Seamless Integration:** Easily integrate with existing GitHub workflows to run evaluation based on rules that you specify in your workflows (e.g., when changes are committed to feature flag configuration or system prompt files).

## Inputs

| Name                               | Description                                                                                                                                             |
| :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| azure-ai-project-connection-string | Connection string of your Azure AI Project                                                                                                              |
| agent-ids                          | Id of the agent(s) to evaluate. If multiple are provided, all agents will be evaluated and compared against the baseline with statistical test results. |
| data-path                          | Path to the data file that contains the evaluators and input for evaluations                                                                            |

Here is a sample data file.

```JSON
{
  "name": "my-test-data",
  "evaluators": ["FluencyEvaluator", "ViolenceEvaluator"],
  "data": [
    {
      "query": "Tell me about Tokyo?",
      "ground_truth": "Tokyo is the capital of Japan and the largest city in the country. It is located on the eastern coast of Honshu, the largest of Japan's four main islands."
    },
    {
      "query": "Where is Italy?",
      "ground_truth": "Italy is a country in southern Europe, located on the Italian Peninsula and the two largest islands in the Mediterranean Sea, Sicily and Sardinia."
    }
  ]
}
```

## Outputs

TODO - add some explanation, screenshots

## Sample workflow

To use this GitHub Action, add this GitHub Action to your CI/CD workflows and specify the trigger criteria (e.g., on commit).

```
name: 'AI Agent Evaluation'

on:
  workflow_dispatch:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

permissions:
  id-token: write
  contents: read

jobs:
  evaluate:
    runs-on: ubuntu-latest
    env:
      DATA-PATH:
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Azure login using Federated Credentials
        uses: azure/login@v2
        with:
          client-id: ${{ vars.AZURE_CLIENT_ID }}
          tenant-id: ${{ vars.AZURE_TENANT_ID }}
          subscription-id: ${{ vars.AZURE_SUBSCRIPTION_ID }}

      - name: Evaluate AI Agents
        uses: microsoft/ai-agent-eval@v1
        with:
          azure-ai-project-connection-string: ${{ vars.AZURE_AIPROJECT_CONNECTION_STRING }}
          data-path: ${{ github.workspace }}/data/golden-dataset.json
          agent-ids: 'agent-id-1, agent-id-2'
```

## Outputs

Evaluation results will be output to the summary section for each AI Evaluation GitHub Action run under Actions in GitHub.com.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
