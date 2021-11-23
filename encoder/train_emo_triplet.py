import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from encoder.data_objects import IemocapDataset
from encoder.params_model import *
from encoder.emo_models import EmoEncoder


def evaluate(model, loader, loss_fn, device):
    """Evaluates the accuracy and loss of model on given data loader"""

    total, total_loss = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            anchor, a_lengths, pos, pos_lengths, neg, neg_lengths = batch
            anchor = anchor.to(device)
            anchor_lengths = a_lengths.cpu()
            pos = pos.to(device)
            pos_lengths = pos_lengths.cpu()
            neg = neg.to(device)
            neg_lengths = neg_lengths.cpu()
            
            packed_anchor = pack_padded_sequence(anchor, anchor_lengths, batch_first=True, enforce_sorted=False)
            anchor_embeds, _ = model(packed_anchor)
            packed_pos = pack_padded_sequence(pos, pos_lengths, batch_first=True, enforce_sorted=False)
            pos_embeds, _ = model(packed_pos)
            packed_neg = pack_padded_sequence(neg, neg_lengths, batch_first=True, enforce_sorted=False)
            neg_embeds, _ = model(packed_neg)

            # Compute loss
            loss = loss_fn(anchor_embeds, pos_embeds, neg_embeds)
            total_loss += loss.item() * anchor.size(0)
            total += anchor.size(0)

    return {
        'loss': total_loss / total
    }


def collate_fn(batch):
    """Collates a batch of padded variable length inputs"""

    anchor, pos, neg = zip(*batch)

    # get sequence lengths
    anchor_lengths = torch.tensor([x.shape[0] for x in anchor])
    pos_lengths = torch.tensor([x.shape[0] for x in pos])
    neg_lengths = torch.tensor([x.shape[0] for x in neg])

    # padding
    anchor = torch.nn.utils.rnn.pad_sequence(anchor, batch_first=True)
    pos = torch.nn.utils.rnn.pad_sequence(pos, batch_first=True)
    neg = torch.nn.utils.rnn.pad_sequence(neg, batch_first=True)

    return anchor, anchor_lengths, pos, pos_lengths, neg, neg_lengths


def train(run_id: str, epoch: int, learning_rate: float, train_meta_path: Path, dev_meta_path: Path, test_meta_path: Path,
          models_dir: Path, save_every: int, backup_every: int, eval_every: int, force_restart: bool):
    """Trains the EmoEncoder model on IEMOCAP dataset"""

    # Create a dataset and a dataloader
    train_dataset = IemocapDataset(train_meta_path, triplet=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=os.cpu_count() - 1,
        collate_fn=collate_fn
    )
    dev_dataset = IemocapDataset(dev_meta_path, triplet=True)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=os.cpu_count() - 1,
        collate_fn=collate_fn
    )
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model and the optimizer
    model = EmoEncoder(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.TripletMarginLoss().to(device)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")

    step = init_step
    total, total_loss = 0, 0
    for e in range(epoch):
        print(f"Training epoch {e}/{epoch}:")
        for batch in tqdm(train_loader):
            model.train()

            # Forward pass
            anchor, a_lengths, pos, pos_lengths, neg, neg_lengths = batch
            anchor = anchor.to(device)
            anchor_lengths = a_lengths.cpu()
            pos = pos.to(device)
            pos_lengths = pos_lengths.cpu()
            neg = neg.to(device)
            neg_lengths = neg_lengths.cpu()

            packed_anchor = pack_padded_sequence(anchor, anchor_lengths, batch_first=True, enforce_sorted=False)
            anchor_embeds, _ = model(packed_anchor)
            packed_pos = pack_padded_sequence(pos, pos_lengths, batch_first=True, enforce_sorted=False)
            pos_embeds, _ = model(packed_pos)
            packed_neg = pack_padded_sequence(neg, neg_lengths, batch_first=True, enforce_sorted=False)
            neg_embeds, _ = model(packed_neg)

            # Compute loss
            loss = loss_fn(anchor_embeds, pos_embeds, neg_embeds)
            total_loss += loss.item() * anchor.size(0)
            total += anchor.size(0)

            # Backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if eval_every != 0 and step % eval_every == 0:
                mean_loss = total_loss / total
                total, total_loss = 0, 0

                metrics = evaluate(model, dev_loader, loss_fn, device)
                writer.add_scalar('Loss/train', mean_loss, step)
                writer.add_scalar('Loss/dev', metrics['loss'], step)

            # Overwrite the latest version of the model
            if save_every != 0 and step % save_every == 0:
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)

            # Make a backup
            if backup_every != 0 and step % backup_every == 0:
                print("Making a backup (step %d)" % step)
                backup_dir.mkdir(exist_ok=True)
                backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, backup_fpath)
                
            step += 1


if __name__ == "__main__":
    train(
        run_id="triplet_1",
        epoch=1000,
        learning_rate=0.00002,
        train_meta_path=Path("iemocap_meta_train.csv"),
        dev_meta_path=Path("iemocap_meta_dev.csv"),
        test_meta_path=Path("iemocap_meta_test.csv"),
        models_dir=Path("encoder/saved_models/"),
        eval_every=500,
        save_every=5000,
        backup_every=20000,
        force_restart=False
    )

# python -m encoder.train_emo_triplet
