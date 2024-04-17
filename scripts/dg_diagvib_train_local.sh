python3 src/train.py \
    --multirun \
    callbacks=none \
    +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_2 \
    experiment=dg/diagvibsix/diagvibsix_erm \
    seed=0 \
    trainer=cpu \
    trainer.max_epochs=1 \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb=null