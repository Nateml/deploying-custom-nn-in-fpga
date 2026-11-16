import copy
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from data_prep import into_dataloader, prefetched
from models.globals import (
    FOLDER,
    MODEL,
    TEST_MINIBATCH_SIZE,
)
from training.utils import confusion_matrix, print_or_output

# array size of old data
BATCH_ARRAY_SIZE = 200


def resume_path(output_dir, filename):
    """Training state saved after every epoch. If it exists, train_model continues
    from it; delete it to start from scratch. Removed when training finishes."""
    return f"{output_dir}models/{filename}_resume.pt"


def save_resume_state(path, **state):
    # write to a temporary file first, so a job killed mid-save keeps the old state
    torch.save(state, f"{path}.tmp")
    os.replace(f"{path}.tmp", path)


def train_model(
    device,
    d,
    out_file,
    output_dir,
    filename,
    epochs,
    optimizer,
    model,
    initial_model,
    loss_fn,
    minibatch_size,
    early_stopping_min_delta,
    early_stopping_patience,
    n_flops,
):
    # initialize arrays to store the losses and accuracies
    train_accuracies = {f"folder {i}": [] for i in range(FOLDER + 1)}
    train_accuracies_stdev = {f"folder {i}": [] for i in range(FOLDER + 1)}
    train_losses = {f"folder {i}": [] for i in range(FOLDER + 1)}
    train_losses_stdev = {f"folder {i}": [] for i in range(FOLDER + 1)}
    validation_accuracies = {f"folder {i}": [] for i in range(FOLDER + 1)}
    validation_accuracies_stdev = {f"folder {i}": [] for i in range(FOLDER + 1)}
    validation_losses = {f"folder {i}": [] for i in range(FOLDER + 1)}
    validation_losses_stdev = {f"folder {i}": [] for i in range(FOLDER + 1)}
    best_epoch = {f"folder {i}": 0 for i in range(FOLDER + 1)}

    total_n_flops = 0

    save_flops = []
    save_loss = []

    torch.cuda.empty_cache()

    history = (
        train_accuracies,
        train_accuracies_stdev,
        train_losses,
        train_losses_stdev,
        validation_accuracies,
        validation_accuracies_stdev,
        validation_losses,
        validation_losses_stdev,
        best_epoch,
    )

    initial_lr = optimizer.param_groups[0]["lr"]
    final_epoch = epochs
    start_folder, start_epoch, stage = 0, 0, None

    resume_file = resume_path(output_dir, filename)
    if os.path.exists(resume_file):
        state = torch.load(resume_file, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for current, saved in zip(history, state["history"]):
            current.update(saved)
        initial_lr = state["initial_lr"]
        final_epoch = state["final_epoch"]
        total_n_flops = state["total_n_flops"]
        save_flops = state["save_flops"]
        save_loss = state["save_loss"]
        start_folder, start_epoch, stage = (
            state["folder"],
            state["epoch"],
            state["stage"],
        )
        print_or_output(
            f"Resuming from {resume_file}: folder {start_folder}, epoch {start_epoch + 1}",
            d,
            out_file,
        )
        del state

    for current_folder in range(start_folder, FOLDER + 1):

        # adaptive learning rate
        if (
            MODEL == "GWaveNet"
            or MODEL == "GWaveNet2D"
            or MODEL == "GWaveNet2DModified"
        ):
            for g in optimizer.param_groups:
                g["lr"] = initial_lr / (current_folder + 1)

        print_or_output(f"Folder {current_folder}", d, out_file)

        first_epoch = 0
        if stage is not None:
            # resuming in the middle of this stage
            first_epoch = start_epoch
            min_validation_loss = stage["min_validation_loss"]
            early_stopping_counter = stage["early_stopping_counter"]
            best_loss = stage["best_loss"]
            best_accuracy = stage["best_accuracy"]
            best_model_state_dict = stage["best_model_state_dict"]
            stage = None
        else:
            min_validation_loss = np.inf
            early_stopping_counter = 0
            final_epoch = epochs

            if current_folder > 0:
                bestmodel_checkpoint = torch.load(f"{output_dir}models/{filename}.pt")
                bestmodel = model.to(device)
                bestmodel.load_state_dict(bestmodel_checkpoint["model_state_dict"])

        for epoch in range(first_epoch, epochs):
            print_or_output(
                f"Used RAM at the beginning of epoch: {(psutil.virtual_memory().used * 1e-9):.2} Gb",
                d,
                out_file,
            )
            torch.cuda.empty_cache()

            # initialize arrays to store relevant variables for the full epoch instead of just per minibatch
            pred_labels_full_epoch = torch.empty(0)
            y_full_epoch = torch.empty(0)
            loss_full_epoch = torch.empty(0)

            n_samples_train = int(10000 * 0.8)
            # reads the next data file in the background while training on the current one
            train_dataloaders = prefetched(
                lambda f, b: into_dataloader(
                    d,
                    out_file,
                    batch=f"train",
                    batch_n=b,
                    minibatch_size=minibatch_size,
                    current_folder=f,
                ),
                [
                    (f, b)
                    for f in range(0, current_folder + 1)
                    for b in range(int(n_samples_train / BATCH_ARRAY_SIZE))
                ],
            )
            for current_current_folder in range(0, current_folder + 1):

                for batch_n in range(int(n_samples_train / BATCH_ARRAY_SIZE)):
                    print_or_output(
                        f"Train batch {batch_n + 1}/{int(n_samples_train/ BATCH_ARRAY_SIZE)}",
                        d,
                        out_file,
                    )

                    model.train()
                    train_dataloader = next(train_dataloaders)

                    for X, y in train_dataloader:
                        total_n_flops += n_flops * minibatch_size
                        save_flops.append(total_n_flops)

                        X, y = X.to(device), y.to(device)
                        X, y = X.float(), y.float()

                        optimizer.zero_grad()

                        pred = model(X)
                        pred = pred.squeeze(1).float()

                        y_pred = torch.sigmoid(pred)
                        pred_labels = torch.round(y_pred)

                        if not pred_labels_full_epoch.numel():
                            pred_labels_full_epoch = pred_labels
                        else:
                            pred_labels_full_epoch = torch.cat(
                                (pred_labels_full_epoch, pred_labels)
                            )

                        if not y_full_epoch.numel():
                            y_full_epoch = y
                        else:
                            y_full_epoch = torch.cat((y_full_epoch, y))

                        loss = loss_fn(pred, y)
                        save_loss.append(loss.item())
                        if not loss_full_epoch.numel():
                            loss_full_epoch = torch.tensor([loss])
                        else:
                            loss_full_epoch = torch.cat(
                                (loss_full_epoch, torch.tensor([loss]))
                            )

                        loss.backward()
                        optimizer.step()

                        del X, y, pred, y_pred, pred_labels, loss
                        torch.cuda.empty_cache()
                        gc.collect()

                    del train_dataloader
                    torch.cuda.empty_cache()
                    gc.collect()

            train_accuracies[f"folder {current_folder}"].append(
                100
                * torch.mean((pred_labels_full_epoch == y_full_epoch).float()).item()
            )
            train_losses[f"folder {current_folder}"].append(torch.mean(loss_full_epoch))

            train_accuracies_stdev[f"folder {current_folder}"].append(
                100 * torch.std((pred_labels_full_epoch == y_full_epoch).float()).item()
            )
            train_losses_stdev[f"folder {current_folder}"].append(
                torch.std(loss_full_epoch)
            )

            torch.cuda.empty_cache()

            print_or_output(
                f"Used RAM before validation: {(psutil.virtual_memory().used * 1e-9):.2} Gb",
                d,
                out_file,
            )

            # change mode into evaluation mode so the model doesn't learn from the validation set
            model.eval()

            # n_samples_val = 1300
            n_samples_val = int(10000 * 0.1)

            pred_labels_full_epoch = torch.empty(0)
            y_full_epoch = torch.empty(0)
            loss_full_epoch = torch.empty(0)
            probabilities_full_epoch = torch.empty(0)

            validation_dataloaders = prefetched(
                lambda f, b: into_dataloader(
                    d,
                    out_file,
                    batch="validation",
                    batch_n=b,
                    minibatch_size=TEST_MINIBATCH_SIZE,
                    current_folder=f,
                ),
                [
                    (f, b)
                    for f in range(0, current_folder + 1)
                    for b in range(int(n_samples_val / BATCH_ARRAY_SIZE))
                ],
            )
            for current_current_folder in range(0, current_folder + 1):
                for batch_n in range(
                    int(n_samples_val / BATCH_ARRAY_SIZE)
                ):  # validation -----------------------------

                    print_or_output(
                        f"Validation batch {batch_n + 1}/{int(n_samples_val / BATCH_ARRAY_SIZE)}",
                        d,
                        out_file,
                    )

                    with torch.no_grad():
                        validation_dataloader = next(validation_dataloaders)

                        for X, y in validation_dataloader:
                            X, y = X.to(device), y.to(device)
                            X, y = X.float(), y.float()

                            pred = model(X)  # [200, 1]
                            pred = pred.squeeze(1).float()  # [200]
                            probabilities = torch.sigmoid(pred)
                            pred_labels = torch.round(probabilities)
                            loss = loss_fn(pred, y)

                            if not pred_labels_full_epoch.numel():
                                pred_labels_full_epoch = pred_labels
                            else:
                                pred_labels_full_epoch = torch.cat(
                                    (pred_labels_full_epoch, pred_labels)
                                )

                            if not y_full_epoch.numel():
                                y_full_epoch = y
                            else:
                                y_full_epoch = torch.cat((y_full_epoch, y))

                            if not loss_full_epoch.numel():
                                loss_full_epoch = torch.tensor([loss])
                            else:
                                loss_full_epoch = torch.cat(
                                    (loss_full_epoch, torch.tensor([loss]))
                                )

                            if not probabilities_full_epoch.numel():
                                probabilities_full_epoch = probabilities
                            else:
                                probabilities_full_epoch = torch.cat(
                                    (probabilities_full_epoch, probabilities)
                                )

                            del X, y, pred, pred_labels, loss
                            torch.cuda.empty_cache()
                            gc.collect()

                del validation_dataloader
                torch.cuda.empty_cache()
                gc.collect()

            validation_accuracies[f"folder {current_folder}"].append(
                100
                * torch.mean((pred_labels_full_epoch == y_full_epoch).float()).item()
            )
            validation_losses[f"folder {current_folder}"].append(
                torch.mean(loss_full_epoch)
            )

            validation_accuracies_stdev[f"folder {current_folder}"].append(
                100 * torch.std((pred_labels_full_epoch == y_full_epoch).float()).item()
            )
            validation_losses_stdev[f"folder {current_folder}"].append(
                torch.std(loss_full_epoch)
            )

            tap = []
            fap = []

            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]

            for threshold in thresholds:
                tp, tn, fp, fn = confusion_matrix(
                    probabilities_full_epoch, y_full_epoch, threshold=threshold
                )

                if fp:
                    print_or_output(
                        f"Threshold: {threshold}\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\nFAP: {fp/(fp+tn)}",
                        d,
                        out_file,
                    )
                else:
                    print_or_output(
                        f"Threshold: {threshold}\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\nFAP: 0",
                        d,
                        out_file,
                    )

                tap.append(tp / (tp + fn))
                fap.append(fp / (fp + tn))

            print_or_output(
                f"Epoch {epoch+1:>12} | Train accuracy: {train_accuracies[f'folder {current_folder}'][-1]:.2f}% | Train loss: {train_losses[f'folder {current_folder}'][-1]:.>6f} | Validation accuracy: {validation_accuracies[f'folder {current_folder}'][-1]:.2f}% | Validation loss: {validation_losses[f'folder {current_folder}'][-1]:.>6f}",
                d,
                out_file,
            )

            plt.plot(thresholds, fap, label=f"Epoch {epoch+1}")
            plt.xlabel("threshold")
            plt.ylabel("FAP")
            plt.legend()
            plt.savefig(f"{output_dir}roc/{filename}.png")

            if epoch == 0:
                with open(f"{output_dir}roc/{filename}.txt", "w") as f:
                    f.write(f"Epoch {epoch+1}\n{tap}\n{fap}\n")
            else:
                with open(f"{output_dir}roc/{filename}.txt", "a") as f:
                    f.write(f"Epoch {epoch+1}\n{tap}\n{fap}\n")

            # early stopping algorithm
            if validation_losses[f"folder {current_folder}"][-1] < (
                min_validation_loss - early_stopping_min_delta
            ):
                min_validation_loss = validation_losses[f"folder {current_folder}"][-1]
                early_stopping_counter = 0
            elif validation_losses[f"folder {current_folder}"][-1] >= (
                min_validation_loss + early_stopping_min_delta
            ):
                early_stopping_counter += 1
                if early_stopping_counter > early_stopping_patience:
                    final_epoch = epoch + 1
                    break

            # conditions to save best model (early stopping 2)
            if (
                MODEL == "GWaveNet"
                or MODEL == "GWaveNet2D"
                or MODEL == "GWaveNet2DModified"
            ):
                if epoch == 0:
                    best_epoch[f"folder {current_folder}"] = epoch + 1
                    best_loss = np.round(
                        validation_losses[f"folder {current_folder}"][-1], 4
                    )
                    best_accuracy = np.round(
                        validation_accuracies[f"folder {current_folder}"][-1], 4
                    )
                    best_model_state_dict = copy.deepcopy(model.state_dict())

                if validation_losses[f"folder {current_folder}"][-1] < best_loss or (
                    validation_losses[f"folder {current_folder}"][-1] == best_loss
                    and validation_accuracies[f"folder {current_folder}"][-1]
                    > best_accuracy
                ):
                    best_epoch[f"folder {current_folder}"] = epoch + 1
                    best_loss = np.round(
                        validation_losses[f"folder {current_folder}"][-1], 4
                    )
                    best_accuracy = np.round(
                        validation_accuracies[f"folder {current_folder}"][-1], 4
                    )
                    best_model_state_dict = copy.deepcopy(model.state_dict())
            elif (
                MODEL == "GregNet"
                or MODEL == "GregNet2D"
                or MODEL == "GregNet2DModified"
            ):
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_epoch[f"folder {current_folder}"] = epoch + 1
                best_loss = np.round(
                    validation_losses[f"folder {current_folder}"][-1], 4
                )
                best_accuracy = np.round(
                    validation_accuracies[f"folder {current_folder}"][-1], 4
                )

            save_resume_state(
                resume_file,
                folder=current_folder,
                epoch=epoch + 1,
                stage={
                    "min_validation_loss": min_validation_loss,
                    "early_stopping_counter": early_stopping_counter,
                    "best_loss": best_loss,
                    "best_accuracy": best_accuracy,
                    "best_model_state_dict": best_model_state_dict,
                },
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                history=history,
                initial_lr=initial_lr,
                final_epoch=final_epoch,
                total_n_flops=total_n_flops,
                save_flops=save_flops,
                save_loss=save_loss,
            )

            torch.cuda.empty_cache()

        print_or_output(
            "Best epoch: " + str(best_epoch[f"folder {current_folder}"]),
            d,
            out_file,
        )

        model_save_path = f"{output_dir}models/{filename}.pt"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        torch.save(
            {"model_state_dict": best_model_state_dict},
            model_save_path,
        )

        # the next stage starts from the best model saved above
        save_resume_state(
            resume_file,
            folder=current_folder + 1,
            epoch=0,
            stage=None,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            history=history,
            initial_lr=initial_lr,
            final_epoch=final_epoch,
            total_n_flops=total_n_flops,
            save_flops=save_flops,
            save_loss=save_loss,
        )

    # load the model of the best epoch
    bestmodel_checkpoint = torch.load(f"{output_dir}models/{filename}.pt")
    bestmodel = initial_model.to(device)
    bestmodel.load_state_dict(bestmodel_checkpoint["model_state_dict"])

    bestmodel.eval()

    test_pred_labels_full_epoch = torch.empty(0)
    y_full_epoch = torch.empty(0)
    loss_full_epoch = torch.empty(0)
    probabilities_full_epoch = torch.empty(0)

    # n_samples_test = 2200
    n_samples_test = int(10000 * 0.1)

    test_dataloaders = prefetched(
        lambda f, b: into_dataloader(
            d,
            out_file,
            batch="test",
            batch_n=b,
            minibatch_size=TEST_MINIBATCH_SIZE,
            current_folder=f,
        ),
        [
            (f, b)
            for f in range(0, FOLDER + 1)
            for b in range(int(n_samples_test / BATCH_ARRAY_SIZE))
        ],
    )
    for current_folder in range(0, FOLDER + 1):
        for batch_n in range(
            int(n_samples_test / BATCH_ARRAY_SIZE)
        ):  # test -----------------------------

            print_or_output(
                f"Test batch {batch_n + 1}/{int(n_samples_test / BATCH_ARRAY_SIZE)}",
                d,
                out_file,
            )

            with torch.no_grad():
                test_dataloader = next(test_dataloaders)

                for X, y in test_dataloader:
                    X, y = X.to(device), y.to(device)
                    X, y = X.float(), y.float()

                    pred = bestmodel(X)
                    pred = pred.squeeze(1).float()
                    probabilities = torch.sigmoid(pred)
                    pred_labels = torch.round(probabilities)
                    loss = loss_fn(pred, y)

                    if not test_pred_labels_full_epoch.numel():
                        test_pred_labels_full_epoch = pred_labels
                    else:
                        test_pred_labels_full_epoch = torch.cat(
                            (test_pred_labels_full_epoch, pred_labels)
                        )

                    if not y_full_epoch.numel():
                        y_full_epoch = y
                    else:
                        y_full_epoch = torch.cat((y_full_epoch, y))

                    if not loss_full_epoch.numel():
                        loss_full_epoch = torch.tensor([loss])
                    else:
                        loss_full_epoch = torch.cat(
                            (loss_full_epoch, torch.tensor([loss]))
                        )

                    if not probabilities_full_epoch.numel():
                        probabilities_full_epoch = probabilities
                    else:
                        probabilities_full_epoch = torch.cat(
                            (probabilities_full_epoch, probabilities)
                        )

                    del X, y, pred, pred_labels, loss, probabilities
                    torch.cuda.empty_cache()
                    gc.collect()

        del test_dataloader
        torch.cuda.empty_cache()
        gc.collect()

    test_accuracy = (
        100 * torch.mean((test_pred_labels_full_epoch == y_full_epoch).float()).item()
    )
    test_loss = torch.mean(loss_full_epoch)

    print_or_output(
        f"Test accuracy: {test_accuracy:.2f}%, Test loss: {test_loss:.4}",
        d,
        out_file,
    )

    # calculate and print confusion matrix
    tp, tn, fp, fn = confusion_matrix(test_pred_labels_full_epoch, y_full_epoch)
    print_or_output(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}", d, out_file)
    cfm = np.array([[tp, fp], [fn, tn]])

    torch.save(
        bestmodel_checkpoint,
        f"{output_dir}models/{test_accuracy}_{filename}.pt",
    )
    os.remove(f"{output_dir}models/{filename}.pt")
    os.remove(resume_file)

    return (
        train_accuracies,
        train_accuracies_stdev,
        train_losses,
        train_losses_stdev,
        validation_accuracies,
        validation_accuracies_stdev,
        validation_losses,
        validation_losses_stdev,
        test_accuracy,
        final_epoch,
        cfm,
        best_epoch,
        probabilities_full_epoch,
        y_full_epoch,
        save_flops,
        save_loss,
    )
