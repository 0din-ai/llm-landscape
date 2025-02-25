SHELL := /bin/bash
TORCHRUN = torchrun --rdzv-backend=c10d --rdzv_endpoint localhost:0 --nnodes=1 --nproc_per_node=$(NGPU)


# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

#
# Virtual Environment Targets
#
clean:
> rm -f .venv_done

.done_venv: clean
> $(PIP) install -r requirements.txt
> $(PIP) install -e .
> touch $@

######################
NGPU := 8  # number of gpus used in the experiments

.SECONDARY:

direction:
> python -m direction

landscape:
> $(TORCHRUN) -m landscape
> python -m plot

finetuning:
> $(TORCHRUN) -m finetuning
