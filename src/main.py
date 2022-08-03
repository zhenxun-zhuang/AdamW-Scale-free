if __name__ == "__main__":
    import torch
    import torch.nn as nn

    import numpy as np
    import os
    import pickle
    import random

    from load_args import load_args
    from data_loader import data_loader
    from resnet import resnet
    from densenet import densenet
    from train import train
    from evaluate import evaluate

    def main():
        args = load_args()

        # Check the availability of GPU.
        use_cuda = args.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Set the ramdom seed for reproducibility.
        if args.reproducible:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            if device != torch.device("cpu"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Load data, note we will also call the validation set as the test set.
        print('Loading data...')
        dataset = data_loader(dataset_name=args.dataset,
                              dataroot=args.dataroot,
                              batch_size=args.batchsize,
                              val_ratio=(args.val_ratio if args.validation else 0))
        train_loader = dataset[0]
        if args.validation:
            test_loader = dataset[1]
        else:
            test_loader = dataset[2]
        num_classes = dataset[-1]

        # Define the model and the loss function.
        if args.model == 'ResNet20':
            net = resnet(depth=20,
                         num_classes=num_classes,
                         no_batch_norm=args.no_batch_norm)
        elif args.model == 'ResNet44':
            net = resnet(depth=44,
                         num_classes=num_classes,
                         no_batch_norm=args.no_batch_norm)
        elif args.model == 'ResNet56':
            net = resnet(depth=56,
                         num_classes=num_classes,
                         no_batch_norm=args.no_batch_norm)
        elif args.model == 'ResNet110':
            net = resnet(depth=110,
                         num_classes=num_classes,
                         no_batch_norm=args.no_batch_norm)
        elif args.model == 'ResNet218':
            net = resnet(depth=218,
                         num_classes=num_classes,
                         no_batch_norm=args.no_batch_norm)
        elif args.model == 'DenseNetBC100':
            net = densenet(depth=100,
                           growthRate=12,
                           num_classes=num_classes,
                           no_batch_norm=args.no_batch_norm)
        else:
            raise ValueError("Unsupported model {0}.".format(args.dataset))

        init_model_path = f"{args.dataset}_{args.model}_"
        init_model_path += 'NoBN_' if args.no_batch_norm else 'BN_'
        init_model_path += "init_model.pt"
        if os.path.isfile(init_model_path):
            net.load_state_dict(torch.load(init_model_path))
        else:
            torch.save(net.state_dict(), init_model_path)

        net.to(device)
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate the model.
        print("Training...")
        running_stats = train(args, train_loader, test_loader, net,
                              criterion, device)
        all_train_losses, all_train_accuracies = running_stats[:2]
        all_test_losses, all_test_accuracies = running_stats[2:4]
        model_stats = running_stats[-1]

        print("Evaluating...")
        final_train_loss, final_train_accuracy = evaluate(train_loader, net,
                                                          criterion, device,
                                                          args.loss_multiplier)
        final_test_loss, final_test_accuracy = evaluate(test_loader, net,
                                                        criterion, device,
                                                        args.loss_multiplier)

        # Logging results.
        print('Writing the results.')
        if not os.path.exists(args.log_folder):
            os.makedirs(args.log_folder)
        log_name = (f'{args.dataset}_{args.model}_'
                    + ('NoBN_' if args.no_batch_norm else 'BN_')
                    + f'{args.optim_method}_'
                    + ('Eta0_%g_' % (args.eta0))
                    + ('WD_%g_' % (args.weight_decay))
                    + (('Scheduler_%s_' % args.scheduler) if args.scheduler else '')
                    + ('Loss_Mul_%g_' % args.loss_multiplier)
                    + ('Epoch_%d_BatchSize_%d_' % (args.train_epochs, args.batchsize))
                    + ('%s' % ('Validation' if args.validation else 'Test')))
        mode = 'w' if args.validation else 'a'
        with open(os.path.join(args.log_folder, log_name + '.txt'), mode) as f:
            f.write('Training running losses:\n')
            f.write('{0}\n'.format(all_train_losses))
            f.write('Training running accuracies:\n')
            f.write('{0}\n'.format(all_train_accuracies))
            f.write('Final training loss is %g\n' % final_train_loss)
            f.write('Final training accuracy is %g\n' % final_train_accuracy)

            f.write('Test running losses:\n')
            f.write('{0}\n'.format(all_test_losses))
            f.write('Test running accuracies:\n')
            f.write('{0}\n'.format(all_test_accuracies))
            f.write('Final test loss is %g\n' % final_test_loss)
            f.write('Final test accuracy is %g\n' % final_test_accuracy)

        if args.store_stats:
            stats_save_path = os.path.join(args.log_folder, log_name+'.pickle')
            with open(stats_save_path, 'wb') as f_model_stats:
                model_stats_file = {"layer names": [param_name for param_name, _ in net.named_parameters()],
                                    "model stats": model_stats}
                pickle.dump(model_stats_file, f_model_stats, protocol=pickle.HIGHEST_PROTOCOL)

        print('Finished.')

    main()
