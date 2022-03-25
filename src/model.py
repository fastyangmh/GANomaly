#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary
from os.path import isfile


#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        in_chans=project_parameters.in_chans,
        input_height=project_parameters.input_height,
        latent_dim=project_parameters.latent_dim,
        classes=project_parameters.classes,
        generator_feature_dim=project_parameters.generator_feature_dim,
        discriminator_feature_dim=project_parameters.discriminator_feature_dim,
        adversarial_weight=project_parameters.adversarial_weight,
        reconstruction_weight=project_parameters.reconstruction_weight,
        encoding_weight=project_parameters.encoding_weight)
    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


#class
class Encoder(nn.Module):
    def __init__(self, input_height, in_chans, out_chans, latent_dim,
                 add_final_conv) -> None:
        super().__init__()
        assert input_height % 16 == 0, 'input_height has to be a multiple of 16'
        layers = []
        layers.append(
            nn.Conv2d(in_channels=in_chans,
                      out_channels=out_chans,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        input_height, out_chans = input_height / 2, out_chans
        while input_height > 4:
            in_channels = out_chans
            out_channels = out_chans * 2
            layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            out_chans *= 2
            input_height /= 2
        if add_final_conv:
            layers.append(
                nn.Conv2d(in_channels=out_chans,
                          out_channels=latent_dim,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, input_height, in_chans, out_chans, latent_dim) -> None:
        super().__init__()
        assert input_height % 16 == 0, 'input_height has to be a multiple of 16'
        out_chans, target_size = out_chans // 2, 4
        while target_size != input_height:
            out_chans *= 2
            target_size *= 2
        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels=latent_dim,
                               out_channels=out_chans,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_chans))
        layers.append(nn.ReLU(inplace=True))
        target_size = 4
        while target_size < input_height // 2:
            layers.append(
                nn.ConvTranspose2d(in_channels=out_chans,
                                   out_channels=out_chans // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_chans // 2))
            layers.append(nn.ReLU(inplace=True))
            out_chans //= 2
            target_size *= 2
        layers.append(
            nn.ConvTranspose2d(in_channels=out_chans,
                               out_channels=in_chans,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, input_height, in_chans, generator_feature_dim,
                 latent_dim) -> None:
        super().__init__()
        self.encoder1 = Encoder(input_height=input_height,
                                in_chans=in_chans,
                                out_chans=generator_feature_dim,
                                latent_dim=latent_dim,
                                add_final_conv=True)
        self.decoder = Decoder(input_height=input_height,
                               in_chans=in_chans,
                               out_chans=generator_feature_dim,
                               latent_dim=latent_dim)
        self.encoder2 = Encoder(input_height=input_height,
                                in_chans=in_chans,
                                out_chans=generator_feature_dim,
                                latent_dim=latent_dim,
                                add_final_conv=True)

    def forward(self, x):
        latent1 = self.encoder1(x)
        x_hat = self.decoder(latent1)
        latent2 = self.encoder2(x_hat)
        return x_hat, latent1, latent2


class Discriminator(nn.Module):
    def __init__(self, input_height, in_chans, discriminator_feature_dim,
                 latent_dim) -> None:
        super().__init__()
        layers = Encoder(input_height=input_height,
                         in_chans=in_chans,
                         out_chans=discriminator_feature_dim,
                         latent_dim=latent_dim,
                         add_final_conv=True)
        layers = list(layers.layers.children())
        self.extractor = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.activation_function = nn.Sigmoid()

    def forward(self, x):
        features = self.extractor(x)
        y = self.activation_function(self.classifier(features))
        return y, features


class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config, in_chans,
                 input_height, latent_dim, classes, generator_feature_dim,
                 discriminator_feature_dim, adversarial_weight,
                 reconstruction_weight, encoding_weight) -> None:
        super().__init__(optimizers_config=optimizers_config,
                         lr=lr,
                         lr_schedulers_config=lr_schedulers_config)
        self.generator = Generator(input_height=input_height,
                                   in_chans=in_chans,
                                   generator_feature_dim=generator_feature_dim,
                                   latent_dim=latent_dim)
        self.discriminator = Discriminator(
            input_height=input_height,
            in_chans=in_chans,
            discriminator_feature_dim=discriminator_feature_dim,
            latent_dim=latent_dim)
        self.adversarial_weight = adversarial_weight
        self.reconstruction_weight = reconstruction_weight
        self.encoding_weight = encoding_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.classes = classes
        self.stage_index = 0

    def configure_optimizers(self):
        optimizers_g = self.parse_optimizers(
            params=self.generator.parameters())
        optimizers_d = self.parse_optimizers(
            params=self.discriminator.parameters())
        if self.lr_schedulers_config is not None:
            lr_schedulers_g = self.parse_lr_schedulers(optimizers=optimizers_g)
            lr_schedulers_d = self.parse_lr_schedulers(optimizers=optimizers_d)
            return [optimizers_g[0],
                    optimizers_d[0]], [lr_schedulers_g[0], lr_schedulers_d[0]]
        else:
            return [optimizers_g[0], optimizers_d[0]]

    def weights_init(self, module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def forward(self, x):
        x_hat, latent1, latent2 = self.generator(x)
        loss = F.l1_loss(input=latent1, target=latent2, reduction='none')
        loss = loss.mean(dim=(1, 2, 3))
        return loss, x_hat

    def shared_step(self, batch):
        x, _ = batch
        x_hat, latent1, latent2 = self.generator(x)
        prob_x, feat_x = self.discriminator(x)
        return x, x_hat, latent1, latent2, prob_x, feat_x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_hat, latent1, latent2, prob_x, feat_x = self.shared_step(
            batch=batch)
        if optimizer_idx == 0:  # generator
            prob_x_hat, feat_x_hat = self.discriminator(x_hat)
            adv_loss = self.l2_loss(feat_x_hat,
                                    feat_x) * self.adversarial_weight
            con_loss = self.l1_loss(x_hat, x) * self.reconstruction_weight
            enc_loss = self.l2_loss(latent2, latent1) * self.encoding_weight
            g_loss = enc_loss + con_loss + adv_loss
            self.log('train_loss',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            self.log('train_loss_generator',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return g_loss
        if optimizer_idx == 1:  #discriminator
            prob_x_hat, feat_x_hat = self.discriminator(x_hat.detach())
            real_loss = self.bce_loss(prob_x, torch.ones_like(input=prob_x))
            fake_loss = self.bce_loss(prob_x_hat,
                                      torch.zeros_like(input=prob_x_hat))
            d_loss = (real_loss + fake_loss) * 0.5
            self.log('train_loss_discriminator',
                     d_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, latent1, latent2, prob_x, feat_x = self.shared_step(
            batch=batch)
        # generator
        prob_x_hat, feat_x_hat = self.discriminator(x_hat)
        adv_loss = self.l2_loss(feat_x_hat, feat_x) * self.adversarial_weight
        con_loss = self.l1_loss(x_hat, x) * self.reconstruction_weight
        enc_loss = self.l2_loss(latent2, latent1) * self.encoding_weight
        g_loss = enc_loss + con_loss + adv_loss
        # discriminator
        prob_x_hat, feat_x_hat = self.discriminator(x_hat.detach())
        real_loss = self.bce_loss(prob_x, torch.ones_like(input=prob_x))
        fake_loss = self.bce_loss(prob_x_hat,
                                  torch.zeros_like(input=prob_x_hat))
        d_loss = (real_loss + fake_loss) * 0.5
        if d_loss.item() < 1e-5:
            self.discriminator.apply(self.weights_init)
        self.log('val_loss', g_loss)
        self.log('val_loss_generator', g_loss, prog_bar=True)
        self.log('val_loss_discriminator', d_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.forward(x)[0]
        self.log('test_loss', loss)
        loss_step = loss.cpu().data.numpy()
        y_step = y.cpu().data.numpy()
        return {'y': y_step, 'loss': loss_step}

    def calculate_threshold(self, x1, x2):
        #estimate kernel density
        kde1 = gaussian_kde(x1)
        kde2 = gaussian_kde(x2)

        #generate the data
        xmin = min(x1.min(), x2.min())
        xmax = max(x1.max(), x2.max())
        dx = 0.2 * (xmax - xmin)
        xmin -= dx
        xmax += dx
        data = np.linspace(xmin, xmax, len(x1))

        #get density with data
        kde1_x = kde1(data)
        kde2_x = kde2(data)

        #calculate intersect
        idx = np.argwhere(np.diff(np.sign(kde1_x - kde2_x))).flatten()
        return data[idx]

    def calculate_confusion_matrix(self, y, loss):
        normal_score = loss[y == self.classes.index('normal')]
        abnormal_score = loss[y == self.classes.index('abnormal')]
        threshold = self.calculate_threshold(x1=normal_score,
                                             x2=abnormal_score)
        max_accuracy, best_threshold = 0, 0
        for v in threshold:
            y_pred = np.where(loss < v, self.classes.index('normal'),
                              self.classes.index('abnormal'))
            confusion_matrix = pd.DataFrame(metrics.confusion_matrix(
                y_true=y, y_pred=y_pred,
                labels=list(range(len(self.classes)))),
                                            index=self.classes,
                                            columns=self.classes)
            accuracy = np.diagonal(
                confusion_matrix).sum() / confusion_matrix.values.sum()
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_threshold = v
        y_pred = np.where(loss < best_threshold, self.classes.index('normal'),
                          self.classes.index('abnormal'))
        confusion_matrix = pd.DataFrame(metrics.confusion_matrix(
            y_true=y, y_pred=y_pred, labels=list(range(len(self.classes)))),
                                        index=self.classes,
                                        columns=self.classes)
        accuracy = np.diagonal(
            confusion_matrix).sum() / confusion_matrix.values.sum()
        return confusion_matrix, accuracy, best_threshold

    def test_epoch_end(self, test_outs):
        stages = ['train', 'val', 'test']
        print('\ntest the {} dataset'.format(stages[self.stage_index]))
        print('the {} dataset confusion matrix:'.format(
            stages[self.stage_index]))
        y = np.concatenate([v['y'] for v in test_outs])
        loss = np.concatenate([v['loss'] for v in test_outs])
        figure = plt.figure(figsize=[11.2, 6.3])
        plt.title(stages[self.stage_index])
        for idx, v in enumerate(self.classes):
            score = loss[y == idx]
            sns.kdeplot(score, label=v)
        plt.xlabel(xlabel='Loss')
        plt.legend()
        plt.close()
        self.logger.experiment.add_figure(
            '{} loss density'.format(stages[self.stage_index]), figure,
            self.current_epoch)
        if stages[self.stage_index] == 'test':
            confusion_matrix, accuracy, best_threshold = self.calculate_confusion_matrix(
                y=y, loss=loss)
            print(confusion_matrix)
            plt.figure(figsize=[11.2, 6.3])
            plt.title('{}\nthreshold: {}\naccuracy: {}'.format(
                stages[self.stage_index], best_threshold, accuracy))
            figure = sns.heatmap(data=confusion_matrix,
                                 cmap='Spectral',
                                 annot=True,
                                 fmt='g').get_figure()
            plt.yticks(rotation=0)
            plt.ylabel(ylabel='Actual class')
            plt.xlabel(xlabel='Predicted class')
            plt.close()
            self.logger.experiment.add_figure(
                '{} confusion matrix'.format(stages[self.stage_index]), figure,
                self.current_epoch)
            self.log('test_accuracy', accuracy)
        self.stage_index += 1


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans,
                        project_parameters.input_height,
                        project_parameters.input_height),
            device='cpu')

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.input_height,
                   project_parameters.input_height)

    # get model output
    loss, x_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(x_hat.shape))
