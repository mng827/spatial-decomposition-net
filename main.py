import os

from data_loader import load_pytorch
from misc.config import process_config
from misc.utils import get_logger, get_args, makedirs
from models.sdnet import SDNet
from train import Trainer


def main():

    config = None
    try:
        args = get_args()
        config = process_config(args.config)

        if config is None:
            raise Exception()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    logger = get_logger('log', logpath=config.summary_dir, filepath=os.path.abspath(__file__))

    train_labelled_data_loader, train_unlabelled_data_loader, test_loader = load_pytorch(config)

    model = SDNet(config.image_size, config.num_anatomical_factors, config.num_modality_factors, config.num_classes)
    print(model)
    trainer = Trainer(model, train_labelled_data_loader, train_unlabelled_data_loader, test_loader, config, logger)

    if config.train:
        trainer.train()

    if config.validation:
        trainer.resume(os.path.join(config.checkpoint_dir, 'model.pth'))

        trainer.test_epoch(debug=False)


if __name__ == "__main__":
    main()

