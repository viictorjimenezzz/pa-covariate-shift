python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_val_modelselection \
    +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_2 \
    experiment=dg/diagvibsix/diagvibsix_lisa \
    model.ppred=0.5 \
    seed=0 \
    trainer=cpu \
    trainer.max_epochs=1 \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb=null