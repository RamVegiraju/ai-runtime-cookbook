# Databricks notebook source
# MAGIC %md
# MAGIC # Compare the base model on a fixed prompt
# MAGIC Run on AI v6 serverless GPU. This is a qualitative check; the held-out
# MAGIC loss comparison in `train.py` remains the quantitative demonstration.
# COMMAND ----------

# ruff: noqa: F821  # dbutils is injected by the Databricks notebook runtime.
import json
from pathlib import Path

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

dbutils.widgets.text("training_run_id", "")
training_run_id = dbutils.widgets.get("training_run_id").strip()
if not training_run_id:
    raise ValueError("Set training_run_id to the Lakeflow training MLflow run ID.")

params = yaml.safe_load((Path.cwd() / "workload.yaml").read_text(encoding="utf-8"))["parameters"]
model_name = params["model_name"]
prompt = (
    "In machine learning, what does LoRA change during fine-tuning, "
    "and what stays frozen? Answer in one sentence."
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
model.eval()
formatted = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
answer = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

training_run = mlflow.get_run(training_run_id)
with mlflow.start_run(experiment_id=training_run.info.experiment_id, run_name="base-model-prompt-check"):
    mlflow.set_tags({"source_training_run_id": training_run_id, "model_role": "base"})
    mlflow.log_params({"base_model": model_name, "max_new_tokens": 100, "do_sample": False})
    mlflow.log_text(prompt, "prompt.txt")
    mlflow.log_text(answer, "base_response.txt")

dbutils.notebook.exit(json.dumps({"model": model_name, "prompt": prompt, "base_response": answer}))
