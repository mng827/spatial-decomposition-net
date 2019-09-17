import torch
import torch.optim as optim

from tqdm import tqdm
import os
import numpy as np
import itertools

import losses
import image_utils

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, train_labelled_data_loader, train_unlabelled_data_loader, test_loader, config, logger):
        self.model = model
        self.train_labelled_data_loader = train_labelled_data_loader
        self.train_unlabelled_data_loader = train_unlabelled_data_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger

        self.model.to(device)
        self.epoch = 0
        self.global_step = 0

        self.unlabelled_labelled_ratio = len(self.train_unlabelled_data_loader) // len(self.train_labelled_data_loader)
        if self.unlabelled_labelled_ratio < 1:
            self.unlabelled_labelled_ratio = 1

        self.fixed_image, self.fixed_gt, _, _, _ = iter(self.test_loader).next()

        self.fixed_image = torch.Tensor(self.fixed_image).to(device)
        self.fixed_gt = torch.LongTensor(self.fixed_gt).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), config.learning_rate, [config.beta1, config.beta2])

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every_itr, gamma=0.1)

    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))

            # for _ in range(100):
            self.train_epoch_labelled()
            self.train_epoch_unlabelled()

            if (cur_epoch + 1) % self.config.save_interval == 0:
                self.test_epoch(debug=False)

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.config.checkpoint_dir, 'model.pth')
        torch.save(save_state, save_name)
        print('Saved model: {}'.format(save_name))

    def resume(self, path):
        save_state = torch.load(path)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.model.load_state_dict(save_state['state_dict'])
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])
        print('Loaded model: {}'.format(path))

    def train_epoch_labelled(self):
        loss_list = []
        dice_list = []

        self.model.train()

        modality_prior = torch.distributions.Normal(loc=0.0, scale=1.0)

        repeated_data_loader = itertools.chain.from_iterable([self.train_labelled_data_loader] *
                                                             self.unlabelled_labelled_ratio * 10)

        for itr, (x, gt, _, _, _) in enumerate(tqdm(repeated_data_loader)):
            x = torch.Tensor(x).to(device)

            gt = torch.LongTensor(gt).to(device)

            anatomical_factor, modality_factor, reconstruction, segmentation_logits = self.model(x)     # NCHW

            ce_loss = losses.softmax_cross_entropy_with_logits(segmentation_logits, gt.squeeze(1))

            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(modality_factor, modality_prior))
            reconstruction_loss = torch.mean(torch.abs(x - reconstruction))

            modality_factor_sample = modality_factor.sample()
            reconstruction_sample = self.model.image_decoder(anatomical_factor, modality_factor_sample)
            modality_factor_recon = self.model.modality_encoder(torch.cat([reconstruction_sample, anatomical_factor],
                                                                          dim=1))

            modality_factor_recon_loss = torch.mean(torch.abs(modality_factor_sample - modality_factor_recon.mean))

            total_loss = self.config.ce_coef * ce_loss + self.config.kl_coef * kl_loss + \
                         self.config.reconstruction_coef * reconstruction_loss + \
                         self.config.modality_factor_recon_coef * modality_factor_recon_loss

            total_loss.backward()
            self.global_step += 1
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(total_loss.cpu().data.numpy())

            prob = torch.nn.Softmax(dim=1)(segmentation_logits)

            dice = image_utils.np_categorical_dice(np.argmax(prob.cpu().data.numpy(), axis=1),
                                                   np.squeeze(gt.cpu().data.numpy(), axis=1),
                                                   self.config.num_classes, axis=(0,1,2), smooth_epsilon=None)
            dice_list.append(dice)

        avg_loss = np.mean(loss_list, axis=0)
        avg_dice = np.mean(dice_list, axis=0)

        self.logger.info("train_labelled | loss: {} | dice: {}".format(avg_loss, avg_dice))

    def train_epoch_unlabelled(self):
        loss_list = []
        dice_list = []

        self.model.train()

        modality_prior = torch.distributions.Normal(loc=0.0, scale=1.0)

        for itr, (x, _, _, _, _) in enumerate(tqdm(self.train_unlabelled_data_loader)):
            x = torch.Tensor(x).to(device)

            anatomical_factor, modality_factor, reconstruction, segmentation_logits = self.model(x)     # NCHW

            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(modality_factor, modality_prior))
            reconstruction_loss = torch.mean(torch.abs(x - reconstruction))

            modality_factor_sample = modality_factor.sample()
            reconstruction_sample = self.model.image_decoder(anatomical_factor, modality_factor_sample)
            modality_factor_recon = self.model.modality_encoder(torch.cat([reconstruction_sample, anatomical_factor],
                                                                          dim=1))

            modality_factor_recon_loss = torch.mean(torch.abs(modality_factor_sample - modality_factor_recon.mean))

            total_loss = self.config.kl_coef * kl_loss + \
                         self.config.reconstruction_coef * reconstruction_loss + \
                         self.config.modality_factor_recon_coef * modality_factor_recon_loss

            total_loss.backward()
            self.global_step += 1
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(total_loss.cpu().data.numpy())

        avg_loss = np.mean(loss_list, axis=0)
        avg_dice = np.mean(dice_list, axis=0)

        self.logger.info("train_unlabelled | loss: {} | dice: {}".format(avg_loss, avg_dice))


    def test_epoch(self, debug):
        loss_list = []
        dice_list = []

        self.model.eval()

        modality_prior = torch.distributions.Normal(loc=0.0, scale=1.0)

        for itr, (x, gt, _, _, _) in enumerate(tqdm(self.test_loader)):
            x = torch.Tensor(x).to(device)

            if gt is not None:
                gt = torch.LongTensor(gt).to(device)

            _, modality_factor, reconstruction, segmentation_logits = self.model(x)     # NCHW

            if gt is not None:
                ce_loss = losses.softmax_cross_entropy_with_logits(segmentation_logits, gt.squeeze(1))
            else:
                ce_loss = 0.0

            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(modality_factor, modality_prior))
            reconstruction_loss = torch.mean((x - reconstruction) ** 2)

            total_loss = self.config.ce_coef * ce_loss + self.config.kl_coef * kl_loss + \
                         self.config.reconstruction_coef * reconstruction_loss

            loss_list.append(total_loss.cpu().data.numpy())

            if gt is not None:
                prob = torch.nn.Softmax(dim=1)(segmentation_logits)

                dice = image_utils.np_categorical_dice(np.argmax(prob.cpu().data.numpy(), axis=1),
                                                       np.squeeze(gt.cpu().data.numpy(), axis=1),
                                                       self.config.num_classes, axis=(0,1,2), smooth_epsilon=None)
                dice_list.append(dice)

        avg_loss = np.mean(loss_list, axis=0)
        avg_dice = np.mean(dice_list, axis=0)

        self.logger.info("test | loss: {} | dice: {}".format(avg_loss, avg_dice))

        if debug:
            _, modality_factor, reconstruction, segmentation_logits = self.model(self.fixed_image)

            prob = torch.nn.Softmax(dim=1)(segmentation_logits)

            pred = np.argmax(prob.cpu().data.numpy(), axis=1)

            plt.imshow(self.fixed_image.cpu().data.numpy()[3,0,:,:], cmap='gray')
            im = plt.imshow(pred[3,:,:], cmap='hot', alpha=0.5)
            im.set_clim(0, 3)
            plt.show()
