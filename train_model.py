from EEGconv_HFO import *

if __name__ == '__main__':
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)  # if you are using multi-GPU.
    np.random.seed(2024)
    random.seed(2024)

    parser = argparse.ArgumentParser(description='seizure detection and classification network')
    parser.add_argument('--train_path', type=str, default=r'/home/zhanglong/HFO_github/train/*.csv', help='path to dataset directory')
    parser.add_argument('--val_path', type=str, default=r'/home/zhanglong/HFO_github/validation/*.csv', help='path to dataset directory')
    parser.add_argument('--test_path', type=str, default=r'C:\EEG-project\ISICDM 2024\test\*.csv', help='path to dataset directory')
    parser.add_argument('--save_path', type=str, default='model_conv.pth', help='path to save the model')
    parser.add_argument('--epochs', type=int, default=300, help='trainging epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
    parser.add_argument('--patience', type=int, default=150, help='Number of epochs for learning rate adjustment.')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu')
    args = parser.parse_args()

    # Load Dataset
    ISICDM2024dataset_train = glob.glob(args.train_path)
    ISICDM2024dataset_val = glob.glob(args.val_path)
    train_array, train_label = get_dataset(ISICDM2024dataset_train)
    val_array, val_label = get_dataset(ISICDM2024dataset_val)
    train_dataset = mydataset(train_array, train_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset))
    val_dataset = mydataset(val_array, val_label)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Define Model
    model = Model(num_depth=128).to(args.device)
    #summary(model, torch.tensor(train_dataset[0]).squeeze(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience)
    patience_counter = 0
    best_val_loss = float('inf')
    best_val_score = 0.0

    # Train Process
    for epoch in range(args.epochs):
        print('=' * 105)
        epoch_train_loss, epoch_train_acc = train(model, train_loader, optimizer, criterion, args.device)
        epoch_val_loss, epoch_val_acc, precision, recall, f1, ave_score = validate(model, val_loader, criterion, args.device)

        print(
            f'Epoch {epoch + 1}/{args.epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - '
            f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        print(
            f'Epoch {epoch + 1}/{args.epochs} - Val Recall: {recall:.4f}, Val Precision: {precision:.4f} - '
            f'Val F1-score: {f1:.4f}, Ave Score: {ave_score:.4f}')

        lr_scheduler.step(epoch_val_loss)

        if ave_score > best_val_score:
            best_val_score = ave_score
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping")
                break