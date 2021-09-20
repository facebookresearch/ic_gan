# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock

# MIT License
""" train_fns.py
Functions for the main loop of training different conditional image models
"""
import torch

import utils
import losses


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}

    return train


def GAN_training_function(
    G,
    D,
    GD,
    ema,
    state_dict,
    config,
    sample_conditionings,
    embedded_optimizers=True,
    device="cuda",
    batch_size=0,
):
    def train(x, y=None, features=None):
        if embedded_optimizers:
            G.optim.zero_grad()
            D.optim.zero_grad()
        else:
            GD.optimizer_D.zero_grad()
            GD.optimizer_G.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, batch_size)
        if y is not None:
            y = torch.split(y, batch_size)
        if features is not None:
            f_ = torch.split(features, batch_size)
        else:
            f_ = None
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config["toggle_grads"]:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config["num_D_steps"]):
            # If accumulating gradients, loop multiple times before an optimizer step
            if embedded_optimizers:
                D.optim.zero_grad()
            else:
                GD.optimizer_D.zero_grad()
            for accumulation_index in range(config["num_D_accumulations"]):
                # Sample conditioning for G
                sampled_cond = sample_conditionings()
                labels_g, f_g = None, None
                if features is not None and y is not None:
                    z_, labels_g, f_g = sampled_cond
                elif y is not None:
                    z_, labels_g = sampled_cond
                elif features is not None:
                    z_, f_g = sampled_cond
                # Tensors to device
                if labels_g is not None:
                    labels_g = (
                        labels_g[:batch_size].to(device, non_blocking=True).long()
                    )
                if f_g is not None:
                    f_g = f_g[:batch_size].to(device, non_blocking=True)
                z_ = z_[:batch_size].to(device, non_blocking=True)
                # Obtain discriminator scores
                D_fake, D_real = GD(
                    z_,
                    labels_g,
                    f_g,
                    x[counter],
                    y[counter] if y is not None else None,
                    f_[counter] if f_ is not None else None,
                    train_G=False,
                    split_D=config["split_D"],
                    policy=config["DiffAugment"],
                    DA=config["DA"],
                )

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(
                    config["num_D_accumulations"]
                )
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config["D_ortho"] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print("using modified ortho reg in D")
                utils.ortho(D, config["D_ortho"])

            if embedded_optimizers:
                D.optim.step()
            else:
                GD.optimizer_D.step()

        # Optionally toggle "requires_grad"
        if config["toggle_grads"]:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        if embedded_optimizers:
            G.optim.zero_grad()
        else:
            GD.optimizer_G.zero_grad()

        counter = 0
        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config["num_G_accumulations"]):
            # Sample conditioning for G
            sampled_cond = sample_conditionings()
            labels_g, f_g = None, None
            if features is not None and y is not None:
                z_, labels_g, f_g = sampled_cond
            elif y is not None:
                z_, labels_g = sampled_cond
            elif features is not None:
                z_, f_g = sampled_cond
            # Tensors to device
            if labels_g is not None:
                labels_g = labels_g.to(device, non_blocking=True).long()
            if f_g is not None:
                f_g = f_g.to(device, non_blocking=True)
            z_ = z_.to(device, non_blocking=True)
            # Obtain discriminator scores
            D_fake = GD(
                z_,
                labels_g,
                f_g,
                train_G=True,
                split_D=config["split_D"],
                policy=config["DiffAugment"],
                DA=config["DA"],
            )
            G_loss = losses.generator_loss(D_fake) / float(
                config["num_G_accumulations"]
            )
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config["G_ortho"] > 0.0:
            print(
                "using modified ortho reg in G"
            )  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(
                G,
                config["G_ortho"],
                blacklist=[param for param in G.shared.parameters()],
            )
        if embedded_optimizers:
            G.optim.step()
        else:
            GD.optimizer_G.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config["ema"]:
            ema.update(state_dict["itr"])

        out = {
            "G_loss": float(G_loss.item()),
            "D_loss_real": float(D_loss_real.item()),
            "D_loss_fake": float(D_loss_fake.item()),
        }
        # Return G's loss and the components of D's loss.
        return out

    return train


def save_weights(
    G,
    D,
    G_ema,
    state_dict,
    config,
    experiment_name,
    embedded_optimizers=True,
    G_optim=None,
    D_optim=None,
):
    utils.save_weights(
        G,
        D,
        state_dict,
        config["weights_root"],
        experiment_name,
        None,
        G_ema if config["ema"] else None,
        embedded_optimizers=embedded_optimizers,
        G_optim=G_optim,
        D_optim=D_optim,
    )
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config["num_save_copies"] > 0:
        utils.save_weights(
            G,
            D,
            state_dict,
            config["weights_root"],
            experiment_name,
            "copy%d" % state_dict["save_num"],
            G_ema if config["ema"] else None,
            embedded_optimizers=embedded_optimizers,
            G_optim=G_optim,
            D_optim=D_optim,
        )
        state_dict["save_num"] = (state_dict["save_num"] + 1) % config[
            "num_save_copies"
        ]


""" This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. """


def save_and_sample(
    G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name
):
    utils.save_weights(
        G,
        D,
        state_dict,
        config["weights_root"],
        experiment_name,
        None,
        G_ema if config["ema"] else None,
    )
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config["num_save_copies"] > 0:
        utils.save_weights(
            G,
            D,
            state_dict,
            config["weights_root"],
            experiment_name,
            "copy%d" % state_dict["save_num"],
            G_ema if config["ema"] else None,
        )
        state_dict["save_num"] = (state_dict["save_num"] + 1) % config[
            "num_save_copies"
        ]

    # Accumulate standing statistics?
    if config["accumulate_stats"]:
        utils.accumulate_standing_stats(
            G_ema if config["ema"] and config["use_ema"] else G,
            z_,
            y_,
            config["n_classes"],
            config["num_standing_accumulations"],
        )


""" This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. """


def test(
    G,
    D,
    G_ema,
    z_,
    y_,
    state_dict,
    config,
    sample,
    get_inception_metrics,
    experiment_name,
    test_log,
    loader=None,
    embedded_optimizers=True,
    G_optim=None,
    D_optim=None,
    rank=0,
):
    print("Gathering inception metrics...")
    if config["accumulate_stats"]:
        utils.accumulate_standing_stats(
            G_ema if config["ema"] and config["use_ema"] else G,
            z_,
            y_,
            config["n_classes"],
            config["num_standing_accumulations"],
        )
    if loader is not None:
        IS_mean, IS_std, FID, stratified_FID, prdc_metrics = get_inception_metrics(
            sample, config["num_inception_images"], num_splits=10, loader_ref=loader
        )
    else:
        IS_mean, IS_std, FID, stratified_FID = get_inception_metrics(
            sample, config["num_inception_images"], num_splits=10
        )
    print(
        "Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f"
        % (state_dict["itr"], IS_mean, IS_std, FID)
    )
    # If improved over previous best metric, save approrpiate copy
    if rank == 0:
        if (config["which_best"] == "IS" and IS_mean > state_dict["best_IS"]) or (
            config["which_best"] == "FID" and FID < state_dict["best_FID"]
        ):
            print(
                "%s improved over previous best, saving checkpoint..."
                % config["which_best"]
            )
            utils.save_weights(
                G,
                D,
                state_dict,
                config["weights_root"],
                experiment_name,
                "best%d" % state_dict["save_best_num"],
                G_ema if config["ema"] else None,
                embedded_optimizers=embedded_optimizers,
                G_optim=G_optim,
                D_optim=D_optim,
            )
            state_dict["save_best_num"] = (state_dict["save_best_num"] + 1) % config[
                "num_best_copies"
            ]
        state_dict["best_IS"] = max(state_dict["best_IS"], IS_mean)
        state_dict["best_FID"] = min(state_dict["best_FID"], FID)
        # Log results to file
        test_log.log(
            itr=int(state_dict["itr"]),
            IS_mean=float(IS_mean),
            IS_std=float(IS_std),
            FID=float(FID),
        )
    return IS_mean, FID
