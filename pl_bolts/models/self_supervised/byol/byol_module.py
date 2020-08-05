from typing import List
from collections import OrderedDict

import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
import torchvision

import pl_bolts
from pl_bolts import metrics
from pl_bolts.datamodules import (
    CIFAR10DataModule,
    STL10DataModule,
    ImagenetDataModule,
)
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.metrics import mean
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from pl_bolts.models.self_supervised.simclr.simclr_transforms import (
    SimCLREvalDataTransform,
    SimCLRTrainDataTransform,
)
from pl_bolts.optimizers.layer_adaptive_scaling import LARS


def cosine_distance(prediction, target):
    norm_p, norm_t = torch.norm(prediction, dim=-1), torch.norm(target, dim=-1)
    return (prediction * target).sum(-1) / (norm_p * norm_t)


def cosine_distance_loss(prediction, target):
    loss = -2 * cosine_distance(prediction, target)
    return loss.mean()


def std_hinge_loss(output, margin):
    (
        projection1,
        projection2,
        prediction1,
        prediction2,
        target1,
        target2,
    ) = output

    loss = F.relu(margin - prediction1.std(0).mean())
    loss += F.relu(margin - prediction2.std(0).mean())

    return loss.mean()


class MLP(nn.Module):
    def __init__(self, mlp_type, dims, eps=None):
        super().__init__()
        layers = OrderedDict()
        type_count = {"l": 0, "b": 0, "r": 0}
        self.dims = dims
        self.output_dim = dims[-1]
        dim_index = 0
        for layer_char in mlp_type:
            if layer_char == "l":
                type_count["l"] += 1
                layers["fc" + str(type_count["l"])] = nn.Linear(
                    dims[dim_index], dims[dim_index + 1]
                )
                dim_index += 1
            elif layer_char == "b":
                type_count["b"] += 1
                layers["bn" + str(type_count["b"])] = nn.BatchNorm1d(
                    dims[dim_index]
                )
            elif layer_char == "r":
                type_count["r"] += 1
                layers["relu" + str(type_count["r"])] = nn.ReLU(inplace=True)
        if eps is not None:
            layers["bn" + str(type_count["b"])] = nn.BatchNorm1d(
                layers["bn" + str(type_count["b"])].num_features, eps=eps
            )

        self.mlp = nn.Sequential(layers)

    def forward(self, x):
        x = self.mlp(x)
        return x


class BYOL(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule = None,
        data_dir: str = "./",
        learning_rate: float = 0.00006,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 128,
        online_ft: bool = False,
        num_workers: int = 4,
        optimizer: str = "lars",
        lars_momentum: float = 0.9,
        lars_eta: float = 0.001,
        loss_temperature: float = 0.5,
        encoder_type: str = "resnet50",
        projector_type: str = "lbrl",
        predictor_type: str = "lbrl",
        start_warmup: float = 0.0,
        base_lr: float = 0.2,
        warmup_epochs: int = 10,
        final_lr: float = 0.0,
        projector_dims: List[int] = [2048, 4096, 256],
        predictor_dims: List[int] = [256, 4096, 256],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.online_evaluator = online_ft

        # init default datamodule
        if datamodule is None:
            datamodule = CIFAR10DataModule(
                data_dir, num_workers=num_workers, batch_size=batch_size
            )
            datamodule.train_transforms = SimCLRTrainDataTransform(
                input_height
            )
            datamodule.val_transforms = SimCLREvalDataTransform(input_height)

        self.datamodule = datamodule

        self.loss_func = self.init_loss()
        self.encoder = self.init_encoder()
        self.projector = self.init_projector()
        self.predictor = self.init_predictor()

        if self.online_evaluator:
            z_dim = self.projector.output_dim
            num_classes = self.datamodule.num_classes
            self.non_linear_evaluator = SSLEvaluator(
                n_input=z_dim, n_classes=num_classes, p=0.2, n_hidden=1024
            )

    def init_loss(self):
        return cosine_distance_loss

    def init_encoder(self):
        encoder = getattr(torchvision.models, self.hparams.encoder_type)()
        encoder.fc = nn.Identity()
        return encoder

    def init_projector(self):
        return MLP(self.hparams.projector_type, self.hparams.projector_dims)

    def init_predictor(self):
        return MLP(self.hparams.predictor_type, self.hparams.predictor_dims)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = self.predictor(z)
        return z

    def forward_target(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z


    def training_step(self, batch, batch_idx):
        if isinstance(self.datamodule, STL10DataModule):
            labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), y = batch
        prediction1 = self.forward(img_1)
        prediction2 = self.forward(img_2)

        with torch.no_grad():
            target2 = self.forward_target(img_2)
            target1 = self.forward_target(img_1)

        loss = self.loss_func(prediction1, target1)
        loss += self.loss_func(prediction2, target2)
        log = {"train_ntx_loss": loss}

        # don't use the training signal, just finetune the MLP to see how we're doing downstream
        if self.online_evaluator:
            if isinstance(self.datamodule, STL10DataModule):
                (img_1, img_2), y = labeled_batch

            with torch.no_grad():
                z1 = self.forward(img_1)

            # just in case... no grads into unsupervised part!
            z_in = z1.detach()

            z_in = z_in.reshape(z_in.size(0), -1)
            mlp_preds = self.non_linear_evaluator(z_in)
            mlp_loss = F.cross_entropy(mlp_preds, y)
            loss = loss + mlp_loss
            log["train_mlp_loss"] = mlp_loss

        result = {"loss": loss, "log": log}

        return result

    def validation_step(self, batch, batch_idx):
        if isinstance(self.datamodule, STL10DataModule):
            labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), y = batch
        prediction1 = self.forward(img_1)
        prediction2 = self.forward(img_2)

        target2 = self.forward_target(img_2)
        target1 = self.forward_target(img_1)

        loss = self.loss_func(prediction1, target1)
        loss += self.loss_func(prediction2, target2)
        result = {"val_loss": loss}

        if self.online_evaluator:
            if isinstance(self.datamodule, STL10DataModule):
                (img_1, img_2), y = labeled_batch
                z1 = self.forward(img_1)

            z_in = z1.reshape(z1.size(0), -1)
            mlp_preds = self.non_linear_evaluator(z_in)
            mlp_loss = F.cross_entropy(mlp_preds, y)
            acc = metrics.accuracy(mlp_preds, y)
            result["mlp_acc"] = acc
            result["mlp_loss"] = mlp_loss

        return result

    def validation_epoch_end(self, outputs: list):
        val_loss = mean(outputs, "val_loss")

        log = dict(val_loss=val_loss,)

        progress_bar = {}
        if self.online_evaluator:
            mlp_acc = mean(outputs, "mlp_acc")
            mlp_loss = mean(outputs, "mlp_loss")
            log["val_mlp_acc"] = mlp_acc
            log["val_mlp_loss"] = mlp_loss
            progress_bar["val_acc"] = mlp_acc

        return dict(val_loss=val_loss, log=log, progress_bar=progress_bar)

    def configure_optimizers(self):
        epoch_len = len(self.datamodule.train_dataloader())
        warmup_lr_schedule = np.linspace(
            self.hparams.start_warmup,
            self.hparams.base_lr,
            epoch_len * self.hparams.warmup_epochs,
        )

        def get_lr(global_step):
            if global_step < len(warmup_lr_schedule):
                return warmup_lr_schedule[global_step]
            else:
                return self.hparams.final_lr + 0.5 * (
                    self.hparams.base_lr - self.hparams.final_lr
                ) * (
                    1
                    + np.cos(
                        np.pi
                        * global_step
                        / (
                            epoch_len
                            * (
                                self.hparams.max_epochs
                                - self.hparams.warmup_epochs
                            )
                        )
                    )
                )

        self.get_lr = get_lr

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "lars":
            optimizer = LARS(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.lars_momentum,
                weight_decay=self.hparams.weight_decay,
                eta=self.hparams.lars_eta,
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.hparams.optimizer}")
        return [optimizer]

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_i,
        second_order_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        print(self.trainer.global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.trainer.global_step)
        optimizer.step()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--online_ft", action="store_true", help="run online finetuner"
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="cifar10",
            help="cifar10, imagenet2012, stl10",
        )

        (args, _) = parser.parse_known_args()
        # Data
        parser.add_argument("--data_dir", type=str, default=".")

        # Training
        parser.add_argument(
            "--optimizer", choices=["adam", "lars"], default="lars"
        )
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--learning_rate", type=float, default=1.0)
        parser.add_argument("--lars_momentum", type=float, default=0.9)
        parser.add_argument("--lars_eta", type=float, default=0.001)
        parser.add_argument(
            "--lr_sched_step", type=float, default=30, help="lr scheduler step"
        )
        parser.add_argument(
            "--lr_sched_gamma",
            type=float,
            default=0.5,
            help="lr scheduler step",
        )
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        # Model
        parser.add_argument("--loss_temperature", type=float, default=0.5)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument(
            "--meta_dir",
            default=".",
            type=str,
            help="path to meta.bin for imagenet",
        )

        return parser


# todo: covert to CLI func and add test
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = BYOL.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    datamodule = None
    if args.dataset == "stl10":
        datamodule = STL10DataModule.from_argparse_args(args)
        datamodule.train_dataloader = datamodule.train_dataloader_mixed
        datamodule.val_dataloader = datamodule.val_dataloader_mixed

        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    elif args.dataset == "imagenet2012":
        datamodule = ImagenetDataModule.from_argparse_args(
            args, image_size=196
        )
        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    model = BYOL(**args.__dict__, datamodule=datamodule)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)


class DensenetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = (
            None  # densenet.densenet121(pretrained=False, num_classes=1)
        )
        del self.model.classifier

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out


class Projection(nn.Module):
    def __init__(self, input_dim=1024, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = 4096
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)
