import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score,roc_auc_score


def train(model, train_dataloader, val_dataloader, criterion, optimizer, ppath, scheduler = None, num_epochs=50, backbone='Retina', save=False, device='cpu', patience=7, train_sampler=None, is_main_process=True):
    # model.to(device)

    if is_main_process:
        named_parameters = dict(model.named_parameters())

        trainable = {
            name: p for name, p in named_parameters.items()
            if p.requires_grad
        }

        print(f"Trainable parameters: {sum(p.numel() for p in trainable.values()):,} / {sum(p.numel() for p in named_parameters.values()):,}")

        print("Trainable model parameters:")
        for name, p in trainable.items():
            print(f"  {name}: shape={tuple(p.shape)}")

        print("Optimizer parameter groups:")
        for i, group in enumerate(optimizer.param_groups):
            group_params = {id(p) for p in group["params"]}
            names = [name for name, p in named_parameters.items() if id(p) in group_params]
            print(f"  group {i}: lr={group['lr']}, "
                f"weight_decay={group['weight_decay']}, names={names}")
            
    binary = True if train_dataloader.dataset.labels.shape[1] == 2 else False

    train_losses = []
    val_losses = []
    # f1_scores = []
    auc_scores = []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        # 'f1_score': 0.0,
        'auc_score': 0.0,
    }
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)

        for batch in tqdm(train_dataloader, total=num_train_batches):
            inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if binary:
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, torch.argmax(labels, dim=1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / num_train_batches
        train_losses.append(avg_train_loss)

        if is_main_process:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                    val_inputs = val_batch['image'].to(device)
                    val_labels = val_batch['labels'].to(device)

                    val_outputs = model(val_inputs)

                    if binary:
                        val_loss += criterion(val_outputs, val_labels.float()).item()
                    else:
                        val_loss += criterion(
                            val_outputs,
                            torch.argmax(val_labels, dim=1)
                        ).item()

                    preds = (
                        val_outputs.round()
                        if binary
                        else torch.argmax(val_outputs, dim=1)
                    )
                    probs = (
                        torch.sigmoid(val_outputs)
                        if binary
                        else torch.softmax(val_outputs, dim=1)
                    )

                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(
                        val_labels.cpu().numpy()
                        if binary
                        else torch.argmax(val_labels, dim=1).cpu().numpy()
                    )

            val_loss /= len(val_dataloader)
            if binary:
                auc = roc_auc_score(all_labels, all_probs, average='macro')
            else:
                auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
            # f1 = f1_score(all_labels, all_preds, average="macro")
            # f1_scores.append(f1)
            auc_scores.append(auc)

            if scheduler is not None:
                scheduler.step(val_loss)
        else:
            # Dummy values for non-rank-0 processes
            val_loss = 0.0
            # f1 = 0.0
            auc = 0.0
        
        # if train_sampler is not None:
        #     torch.distributed.barrier()

        if is_main_process:
            # print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}, F1 Score: {f1}')
            print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}, AUC Score: {auc}')

            # if f1 > best_model_info['f1_score']:
            if auc > best_model_info['auc_score']:
                best_model_info['epoch'] = epoch + 1
                if train_sampler is not None:  # DDP case
                    best_model_info['state_dict'] = {
                        k: v.cpu() for k, v in model.module.state_dict().items()
                    }
                else:
                    best_model_info['state_dict'] = {
                        k: v.cpu() for k, v in model.state_dict().items()
                    }
                # best_model_info['state_dict'] = model.state_dict()
                # best_model_info['f1_score'] = f1
                best_model_info['auc_score'] = auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print('Early stopping triggered.')
                early_stop = True
        else:
            early_stop = False

        if train_sampler is not None:
            stop_tensor = torch.tensor(int(early_stop), device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            early_stop = bool(stop_tensor.item())
        if early_stop:
            break

    if is_main_process and not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        if train_sampler is not None:
            model.module.load_state_dict(best_model_info['state_dict'])
        else:
            model.load_state_dict(best_model_info['state_dict'])

    if is_main_process and save:
        os.makedirs(os.path.join(ppath, 'output/models'), exist_ok=True)
        torch.save(best_model_info['state_dict'], os.path.join(ppath, f'output/models/{backbone}_best.pth'))

    if train_sampler is not None:
        torch.distributed.barrier()

    return model
