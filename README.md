# Navigating the Safety Landscape: Measuring Risks in Finetuning Large Language Models [NeurIPS'24]

[![arxiv badge](https://img.shields.io/badge/arXiv-2405.17374-red)](https://arxiv.org/abs/2405.17374)

You can visualize the safety and capability landscapes of your own LLM!

- Plot the **safety basin** of your own model: if you make small, random tweaks to the model's weights, it stays as safe as the original model within a certain range. However, when these tweaks get large enough, there’s a tipping point where the model’s safety suddenly breaks down.
- Harmful finetuning attacks (HFA) compromise safety by dragging the model away from the safety basin. 
- This safety landscape also shows that the system prompt plays a huge role in keeping the model safe, and that this protection extends to slightly tweaked versions of the model within the safety basin.
- When we test the model’s safety with jailbreaking prompts, we see that these prompts are very sensitive to even small changes in the model's weights.

<p align="center">
<img src="./image/landscape.png" alt="Demo" width="900"/>
</p>

## Research Paper
[**Navigating the Safety Landscape: Measuring Risks in Finetuning Large Language Models**](https://arxiv.org/abs/2405.17374) 

<a href="https://shengyun-peng.github.io/">ShengYun Peng</a><sup>1</sup>,
<a href="https://shengyun-peng.github.io/">Pin-Yu Chen</a><sup>2</sup>,
<a href="https://shengyun-peng.github.io/">Matthew Hull</a><sup>1</sup>,
<a href="https://shengyun-peng.github.io/">Duen Horng Chau</a><sup>1</sup>
<br>
<sup>1</sup>Georgia Tech,
<sup>2</sup>IBM Research

In *NeurIPS 2024*.


## Quick Start
You can plot the 1D and 2D LLM landscapes and compute the VISAGE score for your own models. We are using Llama2-7b-chat as an example. Please modify the yaml file under `/config` for customized experiments. 

### Setup
```bash
make .done_venv
```

### Compute direction
```bash
make direction
```

It consume ~27G on a single A100 GPU. The computed direction is stored at `experiments/advbench/1D_random/llama2/dirs1.pt`.

### Visualize landscape and compute VISAGE score
```bash
make landscape 
```

Change `NGPU` in Makefile to the number of devices on your hardware. 

Change `batch_size` at `config/dataset/default.yaml` to avoid CUDA OOM. 

Model generations are saved at `experiments/advbench/1D_random/llama2/output.jsonl`.

The landscape visualization is saved at `experiments/advbench/1D_random/llama2/1D_random_llama2_landscape.png`.

## Citation
```bibtex
@article{peng2024navigating,
  title={Navigating the Safety Landscape: Measuring Risks in Finetuning Large Language Models},
  author={Peng, ShengYun and Chen, Pin-Yu and Hull, Matthew and Chau, Duen Horng},
  journal={arXiv preprint arXiv:2405.17374},
  year={2024}
}
```

## Contact
If you have any questions, feel free to [open an issue](https://github.com/poloclub/llm-landscape/issues/new) or contact [Anthony Peng](https://shengyun-peng.github.io/).





