#!/bin/bash
conda activate steps_new
export PYTHONPATH=$PYTHONPATH:/home/guy/school/steps/ZSSR/ddtn:/home/dovd/ZSSR/ddtn:/home/guy/school/steps/ZSSR/conv_ae:/home/dovd/ZSSR/conv_ae:/home/guy/school/steps/ZSSR/tf_unet:/home/dovd/ZSSR/tf_unet
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dovd/cuda/lib64/
export PATH=$PATH:/usr/local/cuda/bin/
