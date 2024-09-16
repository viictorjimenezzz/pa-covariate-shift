python3 src/train.py \
    --multirun \
    callbacks=_debug \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_val_modelselection \
    +data/dg/diagvib/modelselection@diagvib_dataset=_debug \
    +data/dg/diagvib/modelselection@diagvib_dataset.dataset_specifications.train=_debug_train \
    +data/dg/diagvib/modelselection@diagvib_dataset.dataset_specifications.test=_hue_test_1749 \
    auxiliary_args.diagvib_task=modelselection \
    model.optimizer._target_=torch.optim.SGD \
    experiment=dg/diagvibsix/diagvibsix_erm \
    model.optimizer.lr=0.0001 \
    data.num_workers=0 \
    seed=0 \
    trainer=cpu \
    trainer.deterministic=true \
    trainer.max_epochs=51 \
    logger.wandb.group=plots \
    logger.wandb.name=erm \
    # logger.wandb=null \
    # +trainer.fast_dev_run=true \
        # logger.wandb=null \
    # +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_2,hue_oodval_2,hue_idval_4,hue_oodval_4 \
    #  data/dg/diagvib/modelselection@diagvib_dataset.dataset_specifications.train=_debug_train \
