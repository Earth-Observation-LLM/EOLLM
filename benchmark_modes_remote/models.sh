# Model roster for the multi-model benchmark-modes sweep.
# Sourced by submit_modes.slurm. One line per model:
#   <hf_repo_id>|<short_name>|<max_model_len>|<conda_env>
#
# short_name      -> results/<short_name>/ and the file prefix.
# max_model_len   -> vLLM context. Thinking is OFF everywhere (letter-only output),
#                    so 16384 is ample for question + a few images. Bump only if a
#                    model's processor expands images to many tokens.
# conda_env       -> which conda env to run THIS model in (optional; defaults to
#                    $ENV / earth_eval). gemma-4-12b uses the `gemma4_unified`
#                    arch, which needs vLLM>0.22 + transformers>=5.12 — the
#                    `vllm_gemma4` env. The Qwen models run in earth_eval (vLLM
#                    0.19.1); they do NOT load under the newer stack reliably, so
#                    we switch envs per model rather than upgrading one env.
#
# All four are AWQ/compressed-tensors quant and fit on the 96GB Blackwell card.
# Order is smallest -> largest so the cheap ones validate the pipeline first.
MODELS=(
  "cyankiwi/Qwen3.5-4B-AWQ-BF16-INT8|qwen3.5-4b-awq|16384|earth_eval"
  "cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8|qwen3.5-9b-awq|16384|earth_eval"
  "cyankiwi/gemma-4-12B-it-AWQ-INT4|gemma-4-12b-awq|16384|vllm_gemma4"
  "Qwen/Qwen2.5-VL-7B-Instruct-AWQ|qwen2.5-vl-7b-awq|16384|earth_eval"
)
