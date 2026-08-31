import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.models import VGGEncoder, Decoder
from utils.utils import ImageFolderDataset, get_transform, adaptive_instance_normalization, calc_mean_std



def parse_arguments() :

    parser = argparse.ArgumentParser(
        description="Train AdaIN Decoder"
    )

    parser.add_argument(
        "--content_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--style_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--vgg",
        type=str,
        default="models/vgg_normalised.pth"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment1"
    )

    parser.add_argument(
        "--final_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--content_size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--style_size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--crop",
        action="store_true"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--lr_decay",
        type=float,
        default=5e-5
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--content_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--style_weight",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=1
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=2
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training on: {device}"
    )

    save_dir = (
        Path("experiments")
        / args.experiment
    )

    save_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.final_size
    )

    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.final_size
    )

    content_dataset = ImageFolderDataset(
        args.content_dir,
        content_transform
    )

    style_dataset = ImageFolderDataset(
        args.style_dir,
        style_transform
    )

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    print(
        "Content batches:",
        len(content_dataloader)
    )

    print(
        "Style batches:",
        len(style_dataloader)
    )

    encoder = VGGEncoder(
        args.vgg
    ).to(device)

    decoder = Decoder().to(device)

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch:
        1.0 /
        (
            1.0
            + args.lr_decay * epoch
        )
    )

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    print("Training...")

    for epoch in range(args.epochs):

        progress_bar = tqdm(
            zip(
                content_dataloader,
                style_dataloader
            ),
            total=min(
                len(content_dataloader),
                len(style_dataloader)
            )
        )

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        for (
            content_batch,
            style_batch
        ) in progress_bar:

            content_batch = (
                content_batch.to(device)
            )

            style_batch = (
                style_batch.to(device)
            )

            c_feats = encoder(
                content_batch
            )

            s_feats = encoder(
                style_batch
            )

            target = adaptive_instance_normalization(
                c_feats[-1],
                s_feats[-1]
            )

            generated = decoder(
                target
            )

            generated_feats = encoder(
                generated
            )

            content_loss = (
                mse_loss(
                    generated_feats[-1],
                    target
                )
                * args.content_weight
            )

            style_loss = 0.0

            for (
                generated_feat,
                style_feat
            ) in zip(
                generated_feats,
                s_feats
            ):

                generated_mean, generated_std = (
                    calc_mean_std(
                        generated_feat
                    )
                )

                style_mean, style_std = (
                    calc_mean_std(
                        style_feat
                    )
                )

                style_loss += (
                    mse_loss(
                        generated_mean,
                        style_mean
                    )
                    +
                    mse_loss(
                        generated_std,
                        style_std
                    )
                )

            style_loss *= args.style_weight

            total_loss = (
                content_loss
                + style_loss
            )

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            running_loss += (
                total_loss.item()
            )

            running_closs += (
                content_loss.item()
            )

            running_sloss += (
                style_loss.item()
            )

            progress_bar.set_description(
                f"Loss: {total_loss.item():.4f} | "
                f"Content: {content_loss.item():.4f} | "
                f"Style: {style_loss.item():.4f}"
            )

        scheduler.step()

        batches = min(
            len(content_dataloader),
            len(style_dataloader)
        )

        running_loss /= batches
        running_closs /= batches
        running_sloss /= batches

        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {running_loss:.4f} | "
            f"Content: {running_closs:.4f} | "
            f"Style: {running_sloss:.4f}"
        )

        if (
            (epoch + 1)
            % args.save_interval
            == 0
        ):

            decoder_path = (
                save_dir
                / f"decoder_{epoch + 1}.pth"
            )

            optimizer_path = (
                save_dir
                / f"optimizer_{epoch + 1}.pth"
            )

            torch.save(
                decoder.state_dict(),
                decoder_path
            )

            torch.save(
                optimizer.state_dict(),
                optimizer_path
            )

            output = torch.cat(
                [
                    content_batch,
                    style_batch,
                    generated
                ],
                dim=0
            )

            save_image(
                output,
                save_dir
                / f"output_{epoch + 1}.png",
                nrow=args.batch_size
            )

    # Save final decoder
    final_decoder_path = Path(
        "models"
    ) / "decoder_final.pth"

    torch.save(
        decoder.state_dict(),
        final_decoder_path
    )

    print(
        f"\nFinal decoder saved to: "
        f"{final_decoder_path}"
    )


if __name__ == "__main__":
    main()