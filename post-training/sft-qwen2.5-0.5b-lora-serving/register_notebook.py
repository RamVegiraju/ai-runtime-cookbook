# Databricks notebook source
# MAGIC %md
# MAGIC # Register a direct AI Runtime training run
# MAGIC Run this notebook on Serverless GPU / A10 / AI v6 after a successful
# MAGIC `databricks air run` training attempt.
# COMMAND ----------

# MAGIC %pip install ninja==1.11.1.4

# COMMAND ----------

# ruff: noqa: F821  # dbutils is injected by the Databricks notebook runtime.
from register import register_model

dbutils.widgets.text("output_root", "/Volumes/main/default/air_examples/sft-qwen2.5-0.5b-lora-serving")
dbutils.widgets.text("training_run_id", "")
dbutils.widgets.text("uc_catalog", "main")
dbutils.widgets.text("uc_schema", "default")
dbutils.widgets.text("uc_model_name", "qwen2_5_0_5b_sft")

register_model(
    output_root=dbutils.widgets.get("output_root"),
    training_run_id=dbutils.widgets.get("training_run_id"),
    catalog=dbutils.widgets.get("uc_catalog"),
    schema=dbutils.widgets.get("uc_schema"),
    model_name=dbutils.widgets.get("uc_model_name"),
)
