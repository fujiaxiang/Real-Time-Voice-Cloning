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

    total, correct, total_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            uttid, features, labels, texts, lengths = batch
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.cpu()
            
            packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
            embeds, pred = model(packed_features)

            # Compute loss
            loss = loss_fn(pred, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze().to(device)).sum().item()

    return {
        'accuracy': 100 * correct / total,
        'loss': total_loss / total
    }


def collate_fn(batch):
    """Collates a batch of padded variable length inputs"""

    ids, inputs, labels, texts = zip(*batch)
    labels = torch.tensor(labels)

    # get sequence lengths
    lengths = torch.tensor([x.shape[0] for x in inputs])

    # padding
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)

    return ids, inputs, labels, texts, lengths


def train(run_id: str, epoch: int, train_meta_path: Path, dev_meta_path: Path, test_meta_path: Path, 
          models_dir: Path, save_every: int, backup_every: int, eval_every: int, force_restart: bool):
    """Trains the EmoEncoder model on IEMOCAP dataset"""

    # Create a dataset and a dataloader
    train_dataset = IemocapDataset(train_meta_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=os.cpu_count() - 1,
        collate_fn=collate_fn
    )
    dev_dataset = IemocapDataset(dev_meta_path)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    loss_fn = nn.CrossEntropyLoss().to(device)
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
    for e in range(epoch):
        print(f"Training epoch {e}/{epoch}:")
        for batch in tqdm(train_loader):
            model.train()

            # Forward pass
            uttid, features, labels, texts, lengths = batch
            features = features.to(device)
            labels = labels.to(device)
            # lengths = lengths.to(device)
            lengths = lengths.cpu()  # It appears lengths have to be 1D CPU tensor

            packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
            embeds, pred = model(packed_features)

            # Compute loss
            loss = loss_fn(pred, labels)

            # Backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if eval_every != 0 and step % eval_every == 0:
                metrics = evaluate(model, dev_loader, loss_fn, device)
                writer.add_scalar('Loss/train', loss.item(), step)
                writer.add_scalar('Loss/dev', metrics['loss'], step)
                writer.add_scalar('Accuracy/dev', metrics['accuracy'], step)

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
        run_id="test2",
        epoch=3000,
        train_meta_path=Path("iemocap_meta_train.csv"),
        dev_meta_path=Path("iemocap_meta_dev.csv"),
        test_meta_path=Path("iemocap_meta_test.csv"),
        models_dir=Path("encoder/saved_models/"),
        eval_every=500,
        save_every=5000,
        backup_every=20000,
        force_restart=False
    )

# python -m encoder.train_emo
