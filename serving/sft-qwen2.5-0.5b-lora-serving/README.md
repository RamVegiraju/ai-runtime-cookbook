# Fine-tune and serve Qwen2.5-0.5B with AI Runtime

This recipe takes `Qwen/Qwen2.5-0.5B-Instruct` from AI Runtime fine-tuning through managed MLflow and Unity Catalog to [Custom LLM Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms) with vLLM. It trains on 64 `tatsu-lab/alpaca` examples with 16 held out, using one A10 GPU for at most 16 steps or eight minutes of optimization; it demonstrates the integration rather than large-scale training or an accuracy gain.

## Before you run

- Enable Custom LLM Serving in an AWS workspace with AI Runtime serverless GPU. Model logging requires AI v6, MLflow 3.12 or newer, and `databricks-sdk` 0.102.0 or newer.
- Set the catalog, schema, model, and endpoint names in `databricks.yml`. You need permission to register Unity Catalog models and create Model Serving endpoints. Each job run creates a model version and creates or updates the endpoint.
- Ensure the default `/Volumes/main/default/air_examples/sft-qwen2.5-0.5b-lora-serving` is writable, or set another existing volume in both `databricks.yml` and `workload.yaml`.
- The Alpaca dataset is licensed **CC BY-NC 4.0**. Use data with suitable rights for other purposes.

## Run

From this directory, choose a configured CLI profile and run the two-task Lakeflow Job:

```bash
databricks bundle validate --strict -t dev --profile <PROFILE>
databricks bundle deploy -t dev --profile <PROFILE>
databricks bundle run qwen_sft -t dev --profile <PROFILE>
```

The AI v6 `train` task reads `workload.yaml`, fine-tunes, tests local vLLM inference, logs to MLflow, and registers a model version. The dependent CPU serverless `deploy` task passes that version to a `GPU_MEDIUM` serving endpoint and queries it. The task waits up to 40 minutes for endpoint provisioning.

For a training-only run using the current AI Runtime CLI:

```bash
databricks air run -f workload.yaml --dry-run --profile <PROFILE>
databricks air run -f workload.yaml --watch --profile <PROFILE>
```

This runs `train.py` from the recipe snapshot and saves merged weights. Use the Lakeflow Job for registration and deployment.

## Results

The training MLflow run records parameters, loss metrics, source, a LoRA adapter, and a result summary. Merged weights are at `<output_root>/<MLFLOW_RUN_ID>/merged`. The job creates a linked packaging run, registers `<catalog>.<schema>.<model_name>`, and deploys its new version to `<endpoint_name>` with scale to zero.

Base and fine-tuned loss use the same 16 held-out examples. These losses and inference requests verify the flow, not factual accuracy or a general quality gain. The optional `compare_base.py` notebook logs a base-model response to the same experiment.

The full Alpaca dataset has about 52,000 examples. For a larger subset, change `train_examples`, `eval_examples`, and `max_steps` in `workload.yaml`. The eight-minute cap covers optimization; other steps add wall time.
