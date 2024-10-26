# Navigating the Safety Landscape: Measuring Risks in Finetuning Large Language Models

[![arxiv badge](https://img.shields.io/badge/arXiv-2405.17374-red)](https://arxiv.org/abs/2405.17374)


<p align='left' style="text-align:left;font-size:1.25em;">
<a href="https://shengyun-peng.github.io/">ShengYun Peng</a><sup>1</sup>,&nbsp;
<a href="https://shengyun-peng.github.io/">Pin-Yu Chen</a><sup>2</sup>,&nbsp;
<a href="https://shengyun-peng.github.io/">Matthew Hull</a><sup>1</sup>,&nbsp;
<a href="https://shengyun-peng.github.io/">Duen Horng Chau</a><sup>1</sup>&nbsp;
<br/> 
<sup>1</sup>Georgia Tech&nbsp;&nbsp;&nbsp;<sup>2</sup>IBM Research&nbsp;&nbsp;&nbsp;
<br/> 
<em>NeurIPS, 2024</em>
</p>

<p align="center">
<img src="./image/landscape.png" alt="Demo" width="1000"/>
</p>

Safety alignment is crucial to ensuring that the behaviors of large language models (LLMs) align with human preferences and restrict harmful actions during inference. However, recent studies show that the alignment can be easily compromised by finetuning with only a few adversarially designed training examples. We aim to measure the risks in finetuning LLMs through navigating the LLM safety landscape. We discover a new phenomenon observed universally in the model parameter space of popular open-source LLMs, termed as “safety basin”: randomly perturbing model weights maintains the safety level of the original aligned model in its local neighborhood. Our discovery inspires us to propose the new VISAGE safety metric that measures the safety in LLM finetuning by probing its safety landscape. Visualizing the safety landscape of the aligned model enables us to understand how finetuning compromises safety by dragging the model away from the safety basin. LLM safety landscape also highlights the system prompt’s critical role in protecting a model, and that such protection transfers to its perturbed variants within the safety basin. These observations from our safety landscape research provide new insights for future work on LLM safety community.


## Quick Start
We are using Llama2-7b-chat as an example. Please modify the yaml file under `/config` for customized experiments. 

### Setup
```bash
make .done_venv
```

### Compute direction
```bash
make direction
```

It consume ~27G on a single A100 GPU. The computed direction is stored at `experiments/advbench/1D_random/llama2/dirs1.pt`.

### Plot landscape and compute VISAGE score
```bash
make landscape 
```

Change `NGPU` in Makefile to the number of devices on your hardware. 

Change `batch_size` at `config/dataset/default.yaml` to avoid CUDA OOM. 

Model generations are saved at `experiments/advbench/1D_random/llama2/output.jsonl`.

The landscape plot is saved at `experiments/advbench/1D_random/llama2/1D_random_llama2_landscape.png`.

## Citation
```bibtex
@article{peng2024navigating,
  title={Navigating the Safety Landscape: Measuring Risks in Finetuning Large Language Models},
  author={Peng, ShengYun and Chen, Pin-Yu and Hull, Matthew and Chau, Duen Horng},
  journal={arXiv preprint arXiv:2405.17374},
  year={2024}
}
```





