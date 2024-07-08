# This repository offers scripts for training and evaluating the Woollie -- LLM for Oncology.
## Training
### Training without QLoRA on smaller models (7B, 13B, 33B):

We utilize DistributedDataParallel (DDP) from PyTorch and 
IBM Spectrum LSF to submit jobs. Below is an example script 
that can be customized for SLURM. We use Weights & Biases to track 
training progress.

```bash
#!/bin/bash
set -x

hostname=`hostname`
nnodes=`echo ${LSB_HOSTS} | tr ' ' '\n' | sort | uniq | wc -l`
node_rank=`echo ${LSB_HOSTS} | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR-1}' | grep ${hostname} | cut -d " " -f2`
master_addr=`echo ${LSB_HOSTS} | tr ' ' '\n' | sort | uniq | head -1`
master_port=12345

project_name="woollie-chat_all"
epochs=12
device_batch_size=2
max_length=2048
global_batch_size=64
export WANDB_RUN_ID=${project_name}_bs${global_batch_size}_len${max_length}

torchrun --nproc_per_node 4     \
         --nnodes=${nnodes} --node_rank=${node_rank} --master_addr=${master_addr} --master_port=${master_port} \
     finetune_chat.py \
    --base_model './merged_chat_all_33b/' \
    --data_path './data_chat/combined_medical_chat.json' \
    --output_dir './chat_finetune-'$WANDB_RUN_ID \
    --micro_batch_size $device_batch_size \
    --num_epochs $epochs \
    --cutoff_len $max_length \
    --resume_from_checkpoint True \
    --val_set_size 1500


```
* `nproc_per_node`: Matches the number of GPUs on each node.
* `base_model`: Baseline models trained from the baseline Llama model, stacked atop aligned model data.
* `data_path`: Path that includes the training dataset.
* `cutoff_len`: The maximum token size used for input, with Llama supporting up to 2048 tokens.
* `micro_batch_size`: Batch size per GPU.
* `global_batch_size`: Total batch size across all GPUs.
* `val_set_size`: Sample size of the dataset used for validation.

### Training without using QLoRA on larger model (65B): 

```bash
#!/bin/bash
set -x

hostname=`hostname`
nnodes=`echo ${LSB_HOSTS} | tr ' ' '\n' | sort | uniq | wc -l`
node_rank=`echo ${LSB_HOSTS} | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR-1}' | grep ${hostname} | cut -d " " -f2`
master_addr=`echo ${LSB_HOSTS} | tr ' ' '\n' | sort | uniq | head -1`
master_port=12345

export WANDB_PROJECT=llm
export WANDB_RESUME=auto
project_name="woollie-chat_all_grand_small_qlora_simple-65b"
epochs=3
device_batch_size=1
max_length=2048
global_batch_size=128
export WANDB_RUN_ID=${project_name}_bs${global_batch_size}_len${max_length}

torchrun --nproc_per_node 4     \
         --nnodes=${nnodes} --node_rank=${node_rank} --master_addr=${master_addr} --master_port=${master_port} \
    finetune_qlora_simple.py \
    --base_model './hf-llama-65b' \
    --data_path './data_chat/combined_grand_all_small_chat.json' \
    --output_dir './chat_finetune-'$WANDB_RUN_ID \
    --micro_batch_size $device_batch_size \
    --eval_steps 200 \
    --save_steps 200 \
    --num_epochs $epochs \
    --cutoff_len $max_length \
    --val_set_size 1500
```

* `eval_steps`: Frequency of model performance evaluation against the validation dataset.
* `save_steps`: Frequency of model checkpoints.

## Evaluation 

We employ the [Language Model Evaluation Harness (tag v0.3.0)](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.3.0) to assess model performance against various benchmarks.


```
#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name="hf-llama-30b"
model="../transformers/${model_name}"
tests="genie_test"
#model="../transformers/hf-llama-30b"
#tests="pubmedqa,usmle,medmcqa,genie_test,openbookqa,sciq"
#tests="hellaswag,arc_*,hendrycksTest-*,truthfulqa_*,wikitext,coqa,logiqa"
python     main.py  \
     --model hf-causal-experimental  \
     --model_args pretrained=${model},load_in_8bit=True,use_accelerate=True \
     --tasks ${tests} \
     --write_out \
     --output_base_path ./${model_name}/${tests} \
     --no_cache
```
* `model_args`: Parameters passed to the LLM during evaluation.
* `write_out`: This option saves the logits for each class in a specified folder.
* `tasks`: Specifies the list of benchmarks used for testing.

## Chat with the model 

You can interact with the model via chat by initializing the baseline model alongside the LoRA adapter.

```bash
python ./app.py \
      ./merged_chat_all_33b \
      ./chat_finetune-woollie-chat_all_qlora_simple-65b \
      8bit
```

* In this setup, you provide the baseline model, LoRA adapter, and optionally enable 8-bit quantization to conserve memory. We utilize [Gradio](https://github.com/gradio-app/gradio) for the user interface.

