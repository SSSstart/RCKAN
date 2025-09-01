from utils import generate_reservoir, iteration_reservoir
from utils import EarlyStopping, ReduceLROnPlateau, data_prepare, data_prepare_2, iteration_reservoir_2
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ikan.KAN import KAN as KAN
import os

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = loadmat("data/KS.mat")["data"]
    train_len, test_len = 2000, 1000
    train_data = data[:, :train_len]
    test_data = data[:, train_len:train_len+test_len]
    dim_reservoir = 300; sigma = 0.8; rho = 1.6; sparsity = 0.01;
    dim_system = len(train_data)
    W_in, A = generate_reservoir(dim_reservoir=dim_reservoir, sigma=sigma,
                                 rho=rho, sparsity=sparsity, dim_system=dim_system)
    R, r_last = iteration_reservoir(data=train_data, W_in=W_in, A=A)
    trainloader, valloader, input_size, output_size = data_prepare(train_data, R)

    learning_rate = 0.001
    hidden_size = 512
    model = KAN([input_size, output_size])
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_criterion = nn.MSELoss().to(device)

    min_delta = 1e-7
    min_lr = 2.5e-6
    early_stopping = EarlyStopping(patience=20, min_delta=min_delta, mode="min", start_from_epoch=6)
    reduce_lr = ReduceLROnPlateau(optimizer, factor=0.8, patience=6, min_delta=min_delta, min_lr=min_lr)
    num_epochs = 1
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (input, target) in enumerate(pbar):
                optimizer.zero_grad()
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = loss_criterion(output, target)
                loss.backward()
                optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input, target in valloader:
                input = input.to(device)
                target = target.to(device)
                output = model(input).to(device)
                val_loss += loss_criterion(output, target).item()
            val_loss /= len(valloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train_Loss: {loss}, "
              f"Val_Loss: {val_loss}, lr = {optimizer.param_groups[0]['lr']}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'data/model.pth')
            print('model saved')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        reduce_lr(val_loss)
    if early_stopping.early_stop:
        early_stopping.load_checkpoint(model)
    else:
        model = torch.load('data/model.pth')
    os.remove('data/checkpoint.pt')
    print('model loaded')

    seq_len = 2
    learning_rate = 0.001
    early_stopping = EarlyStopping(patience=2, min_delta=min_delta, mode="min", start_from_epoch=6)
    reduce_lr = ReduceLROnPlateau(optimizer, factor=0.8, patience=2, min_delta=min_delta, min_lr=min_lr)
    num_epochs = 1
    best_val_loss = float('inf')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    trainloader, valloader, input_size, output_size = data_prepare_2(train_data, R, seq_len)
    W_in = torch.tensor(W_in, dtype=torch.float32, requires_grad=False).to(device)
    A = torch.tensor(A, dtype=torch.float32, requires_grad=False).to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        with tqdm(trainloader) as pbar:
            for i, (seq_x, seq_y) in enumerate(pbar):
                optimizer.zero_grad()
                seq_x = seq_x.to(device)
                seq_y = seq_y.to(device)
                batch_size, seq_len, _ = seq_x.shape
                input = seq_x[:, 0, :]
                total_loss = 0.0
                for t in range(seq_len):
                    output = model(input)
                    step_loss = loss_criterion(output, seq_y[:, t, :])
                    total_loss += step_loss
                    input = iteration_reservoir_2(input, output, W_in, A)
                avg_loss = total_loss / seq_len
                avg_loss.backward()
                optimizer.step()
                total_train_loss += avg_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_x, seq_y in valloader:
                seq_x = seq_x.to(device)
                seq_y = seq_y.to(device)
                batch_size, seq_len, _ = seq_x.shape
                input = seq_x[:, 0, :]
                total_val_loss = 0.0
                for t in range(seq_len):
                    output = model(input)
                    total_val_loss += loss_criterion(output, seq_y[:, t, :]).item()
                    input = iteration_reservoir_2(input, output, W_in, A)
                val_loss += total_val_loss / seq_len
        val_loss /= len(valloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train_Loss: {total_train_loss / len(trainloader)}, "
              f"Val_Loss: {val_loss}, "
              f"lr = {optimizer.param_groups[0]['lr']}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'data/model.pth')
            print('model saved')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        reduce_lr(val_loss)

    if early_stopping.early_stop:
        early_stopping.load_checkpoint(model)
    else:
        model = torch.load('data/model.pth')
    os.remove('data/checkpoint.pt')
    print('model loaded')

    # 简单测试
    model.eval()
    model.to('cpu')
    r_2 = r_last.copy()
    for j in range(2, np.shape(r_2)[0] - 2):
        if np.mod(j, 2) == 0:
            r_2[j] = (r_last[j] * r_last[j]).copy()
    test_example = torch.tensor(r_2, dtype=torch.float32)
    output_example = model(test_example).detach().numpy()[0]
    # print('pred: ', output_example, 'true: ', test_data[:, 0])
    # # np.savez('data/result.npz', pred=output_example, true=test_data[:, 0])
    os.remove('data/model.pth')
    print('FINISHED')

