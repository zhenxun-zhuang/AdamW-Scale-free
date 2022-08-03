"""
Train a model on the training set.
"""

from load_optim import load_optim
from evaluate import evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(args, train_loader, test_loader, net, criterion, device):
    """
    Args:
        args: parsed command line arguments.
        train_loader: an iterator over the training set.
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.
        device: using CPU or GPU.

    Outputs:
        All training losses, training accuracies, test losses, and test
        accuracies on each evaluation during training.
    """
    optimizer = load_optim(params=net.parameters(),
                           optim_method=args.optim_method,
                           eta0=args.eta0,
                           weight_decay=args.weight_decay)

    if args.scheduler == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.train_epochs,
                                      eta_min=0,
                                      last_epoch=-1)

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    model_stats = []
    for epoch in range(1, args.train_epochs + 1):
        cur_epoch_model_stats = []
        net.train()
        iteration = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = args.loss_multiplier * criterion(outputs, labels)
            loss.backward()
            if args.optim_method in ['AdamL2', 'AdamW', 'AdamProx']:
                if args.store_stats and iteration % args.store_stats_interval == 0:
                    _, cur_ite_model_stats = optimizer.step(flag_store_stats=True)
                    cur_epoch_model_stats.append(cur_ite_model_stats)
                else:
                    optimizer.step(flag_store_stats=False)
            else:
                optimizer.step()
            iteration += 1
        model_stats.append(cur_epoch_model_stats)

        if args.scheduler == 'Cosine':
            scheduler.step()

        # Evaluate the model on training and validation dataset.
        if epoch % args.eval_interval == 0:
            train_loss, train_accuracy = evaluate(train_loader, net,
                                                  criterion, device,
                                                  args.loss_multiplier)
            all_train_losses.append(train_loss)
            all_train_accuracies.append(train_accuracy)

            test_loss, test_accuracy = evaluate(test_loader, net,
                                                criterion, device,
                                                args.loss_multiplier)
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)

            print('Epoch %d --- ' % (epoch),
                  'train: loss - %g, ' % (train_loss),
                  'accuracy - %g; ' % (train_accuracy),
                  'test: loss - %g, ' % (test_loss),
                  'accuracy - %g' % (test_accuracy))

    return (all_train_losses, all_train_accuracies,
            all_test_losses, all_test_accuracies,
            model_stats)
