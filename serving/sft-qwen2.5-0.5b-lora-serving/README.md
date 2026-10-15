# Fine-tune and serve Qwen2.5-0.5B with AI Runtime

This sample shows how to fine-tune and deploy a small language model on Databricks. We fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on one A10 GPU with AI Runtime, track it with managed MLflow, register it in Unity Catalog, and deploy it with vLLM on a [Custom LLM Serving endpoint](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms). The 64 Alpaca training examples and 16 held-out examples demonstrate the workflow. For a more realistic use case, use task-specific data and evaluate the fine-tuned model against the base model on representative held-out prompts.

## Before you run

- Use an AWS workspace with AI Runtime serverless GPU and Custom LLM Serving. Model logging requires AI v6, MLflow 3.12 or newer, and `databricks-sdk` 0.102.0 or newer.
- Set the catalog, schema, model, and endpoint names in `databricks.yml`. You need permission to register Unity Catalog models and create serving endpoints.
- Make sure `/Volumes/main/default/air_examples/sft-qwen2.5-0.5b-lora-serving` is writable, or change `output_root` in `workload.yaml` to another existing writable volume.
- The Alpaca dataset is licensed **CC BY-NC 4.0**. Use data with suitable rights for other purposes.

## Run

From this directory, choose a configured CLI profile and run the Lakeflow Job:

```bash
databricks bundle validate --strict -t dev --profile <PROFILE>
databricks bundle deploy -t dev --profile <PROFILE>
databricks bundle run qwen_sft -t dev --profile <PROFILE>
```

`databricks.yml` sets the resource names and includes `resources/qwen_sft.job.yml`, which defines the two Lakeflow tasks. `databricks bundle deploy` uploads the recipe files and creates or updates the job; `databricks bundle run` starts it. The GPU task reads `workload.yaml`, trains and tests the merged model with vLLM, then registers it; the CPU task deploys that version and sends a sample request to the endpoint.

To run only the training step with the AI Runtime CLI:

```bash
databricks air run -f workload.yaml --dry-run --profile <PROFILE>
databricks air run -f workload.yaml --watch --profile <PROFILE>
```

This runs `train.py` from the recipe snapshot and saves merged weights. Use the Lakeflow Job to register and deploy the model.

## Results

The training MLflow run records parameters, loss metrics, a LoRA adapter, and a result summary. Merged weights are at `<output_root>/<MLFLOW_RUN_ID>/merged`. The job creates a linked packaging run, registers `<catalog>.<schema>.<model_name>`, and deploys its new version to `<endpoint_name>` with scale to zero.

Base and fine-tuned loss use the same 16 held-out examples. These losses and the sample requests check that the workflow runs; they do not establish factual accuracy or a general quality gain.

The full Alpaca dataset has about 52,000 examples. To use more, adjust the example counts, `max_steps`, `max_length`, batch size, and gradient accumulation in `workload.yaml`. The eight-minute cap applies to optimization only.
