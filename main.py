import argparse
import torch.cuda
import wandb
from util.selector import get_trainer_and_model, get_unet_type
from util.yaml import load_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DTLS with a YAML configuration file.")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        required=True,
        help="Path to the YAML configuration file for training or evaluation",
    )
    args = parser.parse_args()
    config = load_yaml(args.config_path)
    
    if config.wandb_config.use == True:
        run = wandb.init(
            name=config.wandb_config.name,
            project=config.wandb_config.project,
            resume=config.wandb_config.resume,
            id=config.wandb_config.id,
            config=config
        )
    device = config.device if torch.cuda.is_available() else "cpu"
    model = get_unet_type(config.networks.unet_type.refine)
    model = model(**config.networks.refine_unet).to(device)
    trainer, dtls = get_trainer_and_model(config.networks.method)
    
    dtls = dtls(
        model,
        **config.networks.refine_dtls,
        device=device,
    ).to(device)
    
    trainer = trainer(
        dtls,
        **config.trainer,
        device = device,
    )

    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'eval':
        trainer.evaluation()
    elif config.mode == 'single':
        trainer.single_evaluation()
    if config.wandb_config.use == True:    
        wandb.finish()