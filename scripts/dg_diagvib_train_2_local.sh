python3 src/train.py \
    --multirun \
    callbacks=default_train_datashift \
    auxiliary_args.diagvib_task=datashift \
    auxiliary_args.project_name="DiagVib-6 Paper" \
    +callbacks@callbacks.posterioragreement=pametric \
    callbacks.posterioragreement.deltametric=true \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_datashift_val \
    callbacks.posterioragreement.pairing_csv=null \
    callbacks.feature_pairing.method=null \
    callbacks.feature_pairing.index=L2 \
    callbacks.feature_pairing.nearest=false \
    +data/dg/diagvib/datashift@diagvib_dataset=paper \
    experiment=dg/diagvibsix/diagvibsix_erm \
    logger.wandb=null\
    trainer.max_epochs=1 \
    model.optimizer.lr=0.0001 \
    seed=0 \
    trainer=cpu \
    trainer.deterministic=true \
    +trainer.fast_dev_run=true \