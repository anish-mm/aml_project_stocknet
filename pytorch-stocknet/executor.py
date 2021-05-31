import os
import os.path as osp
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import logging
from copy import deepcopy

from model import Stocknet
from stocknet_dataset import StocknetDataset
# from stocknet_dataset_from_file import StocknetDataset

from config_loader import dates, vocab, vocab_size, stock_symbols, path_parser, config_model as cmodel
from metrics import n_accurate, get_true_yT, eval_acc, eval_res

def train_and_dev(
    log_dir="../runs/pytorch-stocknet", 
    dir_checkpoint="../checkpoints/pytorch-stocknet", name_prefix=""
):
    name = f"{name_prefix}_pytorch-stocknet_bs_{cmodel['batch_size']}_lr_{cmodel['lr']}_opt_{cmodel['opt']}"
    log_dir = osp.join(log_dir, name)
    dir_checkpoint = osp.join(dir_checkpoint, name)

    for p in [log_dir, dir_checkpoint]:
        if not osp.exists(p):
            os.makedirs(p)

    current_epoch = 0
    current_best_val_score = 0 # acc
    best_model_state = None
    epoch_of_best_model = -1
    best_saved = True
    # best_model_state, \
    # current_epoch, current_loss, current_best_val_score, epoch_of_best_model, \
    # best_saved

    writer = SummaryWriter(log_dir=log_dir)

    print("Setting up training dataset...")
    train_dataset = StocknetDataset(
        start_date=dates["train_start_date"], end_date=dates["train_end_date"],
        vocab=vocab, vocab_size=vocab_size, stock_symbols=stock_symbols,
        movement_path=path_parser.movement, tweet_path=path_parser.preprocessed,
        word_embed_size=cmodel["word_embed_size"], glove_path=path_parser.glove,
        word_embed_type=cmodel["word_embed_type"], max_n_days=cmodel["max_n_days"],
        max_n_msgs=cmodel["max_n_msgs"], max_n_words=cmodel["max_n_words"],
        y_size=cmodel["y_size"],
    )
    print("Setting up development dataset...")
    dev_dataset = StocknetDataset(
        start_date=dates["dev_start_date"], end_date=dates["dev_end_date"],
        vocab=vocab, vocab_size=vocab_size, stock_symbols=stock_symbols,
        movement_path=path_parser.movement, tweet_path=path_parser.preprocessed,
        word_embed_size=cmodel["word_embed_size"], glove_path=path_parser.glove,
        word_embed_type=cmodel["word_embed_type"], max_n_days=cmodel["max_n_days"],
        max_n_msgs=cmodel["max_n_msgs"], max_n_words=cmodel["max_n_words"],
        y_size=cmodel["y_size"],
    )

    print("Setting up data loaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=cmodel["batch_size"], num_workers=1,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=32, num_workers=1, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Setting up the model...")
    net = Stocknet(
        word_table_init=torch.tensor(train_dataset.init_word_table(), device=device), 
        **cmodel, price_size=3, z_size=cmodel["h_size"],
    )

    print("use if GPU available...")
    net.to(device)

    print("Setting up optimizer...")
    if cmodel["opt"] == "adam":
        optimizer = Adam(net.parameters(), lr=cmodel["lr"])
    # else:
    #     ###  configure according to your needs

    print("Begin Training...")
    global_step, n_samples_global, batch_count = 0, 0, 0
    n_train = train_dataset.n_samples
    n_val = dev_dataset.n_samples
    epochs = cmodel["n_epochs"]
    for epoch in tqdm(range(current_epoch, epochs)):
        net.train()
        epoch_loss, epoch_acc, epoch_n_acc, n_batches, n_trained = 0, 0, 0, 0, 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='sample', position=0, leave=True) as pbar:
            for batch in train_loader:
                batch_count+= 1
                # print(f"\nbatch = {batch_count}")
                this_batch_size = batch["T"].shape[0]
                y_T, loss = net(
                    prices=batch["prices"].to(device), words=batch["msgs"].to(device), 
                    n_days=batch["T"].to(device), n_msgs=batch["n_msgs"].to(device), 
                    n_words=batch["n_words"].to(device), y_true=batch["ys"].to(device), 
                    ss_indices=batch["ss_indices"].to(device), global_step=global_step,
                )

                global_step += 1
                n_samples_global += this_batch_size
                n_batches += 1
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), n_samples_global)

                true_yT = get_true_yT(batch["ys"], batch["T"]).to(device)
                n_acc = n_accurate(y_T, true_yT)
                epoch_n_acc += n_acc
                acc = n_acc / float(this_batch_size)
                epoch_acc += acc
                n_trained += this_batch_size
                print(f"""
                    batch n_acc / batch_size     = {n_acc} / {this_batch_size} 
                    epoch_n_acc / n_trained      = {epoch_n_acc} / {n_trained}
                    n_trained / n_train          = {n_trained} / {n_train}
                """)

                writer.add_scalar('Accuracy/train', acc, n_samples_global)
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'acc (batch)' : acc.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(this_batch_size)
                

                if global_step % ( n_train // (10 * cmodel["batch_size"])) == 0:
                    # validation
                    val_score = 0
                    net.eval()
                    n_acc_val = 0
                    with tqdm(total=n_val, desc='Validation round', unit='sample', leave=True, position=0) as vpbar:
                        n_valed = 0
                        for dev_batch in dev_loader:
                            dev_batch_size = dev_batch["T"].shape[0]
                            n_valed += dev_batch_size
                            with torch.no_grad():
                                y_T = net(
                                    prices=dev_batch["prices"].to(device), words=dev_batch["msgs"].to(device), 
                                    n_days=dev_batch["T"].to(device), n_msgs=dev_batch["n_msgs"].to(device), 
                                    n_words=dev_batch["n_words"].to(device), y_true=dev_batch["ys"].to(device), 
                                    ss_indices=dev_batch["ss_indices"].to(device), 
                                )
                            true_yT = get_true_yT(dev_batch["ys"], dev_batch["T"]).to(device)
                            n_acc_val += n_accurate(y_T, true_yT)
                            vpbar.update(dev_batch_size)
                        vpbar.update(n_val - n_valed)
                    # val_score = n_acc_val / float(n_val)
                    # print(f"val_score = {val_score}")
                    print(f"n_acc_val = {n_acc_val}, n_valed = {n_valed} / {n_val}")
                    val_score = eval_acc(n_acc_val, n_valed)
                    
                    if val_score > current_best_val_score:
                        print(f"""
                        New Best Model!
                        Previous Dev Accuracy = {current_best_val_score}
                        New Dev Accuracy      = {val_score}
                        Improvement           = {val_score - current_best_val_score}
                        """)

                        best_saved = False
                        current_best_val_score = val_score
                        best_model_state = deepcopy(net.state_dict())
                        epoch_of_best_model = epoch + 1
                    
                    print('Validation accuracy: {}'.format(val_score))
                    writer.add_scalar('Accuracy/val', val_score, n_samples_global)
                    net.train()
                pbar.update(n_train - n_samples_global)
        current_epoch = epoch + 1
        epoch_loss = epoch_loss / float(n_batches)
        print(f"Epoch {current_epoch}: trained on {n_trained} / {n_train}")
        epoch_acc = eval_acc(epoch_n_acc, n_trained)
        writer.add_scalar('Accuracy/Epoch-train', epoch_acc, current_epoch)
        writer.add_scalar('Loss/Epoch-train', epoch_loss, current_epoch)

        try:
            os.makedirs(dir_checkpoint)
            print('Created checkpoint directory')
        except OSError:
            pass
    
        torch.save({
        'current_epoch': current_epoch,
        'model_state_dict': net.state_dict(), # model state after the last completed epoch. can be used to resume.
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        'current_best_val_score': current_best_val_score,
        "epoch_of_best_model": epoch_of_best_model,
        }, osp.join(dir_checkpoint, "last.tar"))

        if not best_saved:
            torch.save({
            'model_state_dict': best_model_state,
            # 'scheduler_state_dict': scheduler.state_dict(),
            'current_best_val_score': current_best_val_score,
            "epoch_of_best_model": epoch_of_best_model,
            }, osp.join(dir_checkpoint, "best.tar"))
            
            print(f'Saved new best model with val Accuracy {current_best_val_score}.')
            best_saved = True
    print(f"""
        Traning done...
        Best validation accuracy obtained = {current_best_val_score}
    """)

def test(
    checkpoint_path,
    use_mcc=True,
):
    """
    Load model from given path, and test it on test data.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up test dataset...")
    test_dataset = StocknetDataset(
        start_date=dates["test_start_date"], end_date=dates["test_end_date"],
        vocab=vocab, vocab_size=vocab_size, stock_symbols=stock_symbols,
        movement_path=path_parser.movement, tweet_path=path_parser.preprocessed,
        word_embed_size=cmodel["word_embed_size"], glove_path=path_parser.glove,
        word_embed_type=cmodel["word_embed_type"], max_n_days=cmodel["max_n_days"],
        max_n_msgs=cmodel["max_n_msgs"], max_n_words=cmodel["max_n_words"],
        y_size=cmodel["y_size"],
    )

    print("Setting up data loader...")
    test_loader = DataLoader(
        test_dataset, batch_size=100, num_workers=1,
        pin_memory=True,
    )

    print("Setting up the model...")
    net = Stocknet(
        word_table_init=torch.tensor(test_dataset.init_word_table(), device=device), 
        **cmodel, price_size=3, z_size=cmodel["h_size"],
    )
    net.to(device)
    print("Load pretrained weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print("Beginning evaluation...")
    net.eval()
    num_samples, n_acc_test = 0, 0
    true_ys, pred_ys = list(), list()
    n_test = test_dataset.n_samples
    with tqdm(total=n_test, desc=f'Testing', unit='sample', position=0, leave=True) as pbar:
        for batch in test_loader:
            this_batch_size = batch["T"].shape[0]
            num_samples += this_batch_size

            with torch.no_grad():
                y_T = net(
                    prices=batch["prices"].to(device), words=batch["msgs"].to(device), 
                    n_days=batch["T"].to(device), n_msgs=batch["n_msgs"].to(device), 
                    n_words=batch["n_words"].to(device), y_true=batch["ys"].to(device), 
                    ss_indices=batch["ss_indices"].to(device), 
                )
            true_yT = get_true_yT(batch["ys"], batch["T"]).to(device)
            n_acc_test += n_accurate(y_T, true_yT)
            true_ys.append(true_yT.cpu().detach().numpy())
            pred_ys.append(y_T.cpu().detach().numpy())
            pbar.update(this_batch_size)
        pbar.update(n_test - num_samples)
    
    results = eval_res(n_acc_test, num_samples, pred_ys, true_ys, use_mcc)
    print(f"""
    accuracy : {results["acc"]}
    MCC      : {results["mcc"] if use_mcc else 'Not calculated'}
    """)
    return results









