
respect arguments that are passed in. For example, not respecting topk flag.
optimizers currently don't respect the optimization arguments completely.
right now, args is a hierarchical mess. Issue is if we want automatic batch size setting,
we want hparams.batch_size while rn, we have hparams.optim_args.batch size.
in general, hparams should only really have hyperparameters, and rn there's a lot of useless stuff (that you see on the TB logger)
TB logger the 'metrics' don't work. This might be okay if we move to WandB (which Adriel is currently integrating in another branch.)
add scheduler: linear warmup, warm restarts, cosine scheduling, step.
check uncertainty strategies still work. Can we have unit tests around these?
check custom datasets.
set hparams and log on tb.
getting test and select_ensemble to work.


main task
Are we able to reproduce the chexpert table 3?
bash_scripts folder has a script to run all models neccessary for table 3 generation, maybe check that out.
