#pip install -U tensorboard 

tensorboard dev upload --logdir logs_ddpg \
    --name "Walker 2d test ddpg" \
    --description "testing hparams of walker 2d" 
