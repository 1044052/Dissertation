# encoding=utf-8
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from pKa_net import get_model

import os
from pka_dataset import PkaDatasetCSV
from pka_criterion import criterion, criterion2, criterion3
from evaluate import evaluate_model
from tensorboardX import SummaryWriter
import gc
import time


def train():
    """
    train model
    :return: None
    """
    time1 = time.time()
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')  # if has cuda, use gpu divice, if not, use cpu divice.
    # train dataset relative files
    # Changing this one to my one
    # Old one: '../data/model_input/final_train_data/train_n247_f20_n4.csv'
    train_data_path = '/Users/rachel/Documents/Dissertation/new_train_csv.csv'
    train_center_coors_path = '../data/model_input/final_train_data/CpHMD_pka247_center_coors.csv'
    train_protein_features_path = '../data/model_input/final_train_data/data_pdb_CpHMD247_fixed_mol2.csv'
    # test dataset  relative files
    # Changing this one to my one
    # Old one: '../data/model_input/final_val_data/val_n27_f20_n4.csv'
    test_data_path = '/Users/rachel/Documents/Dissertation/new_test_csv.csv'
    test_center_coors_path = '../data/model_input/final_val_data/CpHMD_pka27_center_coors.csv'
    test_protein_features_path = '../data/model_input/final_val_data/data_pdb_CpHMD27_fixed_mol2.csv'

    batch_size = 32   # mini batch size, how many data forward and backward every time.
    total_epoch = 40
    start_epoch = 0
    radii = 10
    train_info_save_dir = './train_info/model_bn4_s21_elu_n247_f20_r4_zscore_gridcharge_adam_chimera_rotate90'
    train_info_save_path = os.path.join(train_info_save_dir, 'pka_train_info.txt')
    model_save_dir = './model/model_bn4_s21_elu_n247_f20_r4_zscore_gridcharge_adam_chimera_rotate90'
    rotate_angle = 90
    is_rotate = True
    normalize = True
    fill_charge = 'grid charge'             # 'grid charge', 'box charge' or 'atom charge'
    load_weight_path = None
    # load_weight_path = './model/model_s21_relu_n252_f19_r4_atomcharge_adam_chimera/pka_net_epoch469.pt'

    mini_loss = 100000.0
    best_model_name = 'pka_net_best.pt'
    if not os.path.exists(train_info_save_dir):
        os.mkdir(train_info_save_dir)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    # load model
    # Change this!!!
    net = get_model('PkaNetRachelELU')
    if load_weight_path:
        # get best model loss
        best_path = os.path.join(model_save_dir, 'pka_net_best.pt')
        net.load_state_dict(torch.load(best_path), strict=False)
        net.eval()
        mini_loss1, RMSE1, _, _ = evaluate_model(net=net, device=device, test_data_path=test_data_path,
                                             batch_size=batch_size, is_rotate=False, fill_charge=fill_charge,
                                             normalize=normalize, center_coors_path=test_center_coors_path,
                                             protein_features_path=test_protein_features_path, radii=radii)

        # get loaded model loss
        net.load_state_dict(torch.load(load_weight_path), strict=False)
        net.eval()
        mini_loss2, RMSE2, _, _ = evaluate_model(net=net, device=device, test_data_path=test_data_path,
                                             batch_size=batch_size, is_rotate=False, fill_charge=fill_charge,
                                             normalize=normalize, center_coors_path=test_center_coors_path,
                                             protein_features_path=test_protein_features_path, radii=radii)

        # get mini loss
        mini_loss = min(RMSE1, RMSE2)

    # set running device, set optim
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.994)

    writer1 = SummaryWriter(train_info_save_dir)
    # writer1.add_graph(model=PkaNetBN2())
    # because set is_rotate 'True', so every time load a diffrent dataset.
    # (some data possible has be rotated.)
    # Old one: data_path=train_data_path, is_rotate=is_rotate, rotate_angle=rotate_angle,
    #                                   fill_charge=fill_charge, normalize=normalize, center_coors_path=train_center_coors_path,
    #                                   proteins_features_path=train_protein_features_path, radii=radii
    # Old test one
    # is_rotate=False, rotate_angle=rotate_angle,
    #                                  fill_charge=fill_charge, normalize=normalize, center_coors_path=test_center_coors_path,
    #                                  proteins_features_path=test_protein_features_path, radii=radii)
    train_dataset = PkaDatasetCSV(data_path=train_data_path)
    test_dataset = PkaDatasetCSV(data_path=test_data_path)
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(start_epoch, total_epoch):
        running_loss = 0.0
        net.train()  # set train model
        train_dataset.flash_batch_data()
        while True:

            for i, data in enumerate(trainloader):
                # get the inputs: data is a list of [input, labels]
                inputs, labels = data
                pkas = labels[:, :1]
                #print(labels)
                name_idxes = labels[:, 1:]
                inputs = inputs.to(device)
                pkas = pkas.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                try:
                    outputs = net(inputs)
                except Exception as e:
                    print(e)
                    # print(inputs)
                    #print(inputs.shape)
                    exit(0)
                #print(outputs)
                loss = criterion(outputs, pkas)
                #print(loss)
                loss.backward()
                optimizer.step()

                # print statistics
                #print(loss, pkas.shape[0])
                running_loss += loss * pkas.shape[0]
                print('train: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))

            # if there is any remaining batch data, load the remaining batch data.
            if not train_dataset.is_empty():
                train_dataset.batch_load_data()
            else:
                break
        # change lr
        # scheduler.step()
        # write training info in files.
        train_total_len = train_dataset.get_total_len()
        running_loss_train = running_loss / train_total_len
        train_info = 'train: [epoch {}] loss: {}'.format(epoch, running_loss_train).ljust(45, ' ')
        with open(train_info_save_path, 'a') as f:
            f.write(train_info)

        # test model
        test_dataset.flash_batch_data()
        running_loss_test, RMSE_test, R2_test, R2_test_shift = evaluate_model(net=net, device=device,
                                                                              test_data_path=None,
                                                                              test_dataset=test_dataset,
                                                                              batch_size=batch_size,
                                                                              is_rotate=False,
                                                                              rotate_angle=rotate_angle,
                                                                              fill_charge=fill_charge,
                                                                              normalize=normalize,
                                                                              center_coors_path=test_center_coors_path,
                                                                              protein_features_path=test_protein_features_path,
                                                                              radii=radii)
        test_info = 'test: [epoch {}] loss: {}\n'.format(epoch, running_loss_test).ljust(50, ' ')
        with open(train_info_save_path, 'a') as f:
            f.write(test_info)

        # draw plot in tensorboard
        writer1.add_scalars('loss[MAE]', tag_scalar_dict={'train': running_loss_train, 'test': running_loss_test},
                            global_step=epoch)
        writer1.add_scalars('RMSE', tag_scalar_dict={'test': RMSE_test}, global_step=epoch)
        writer1.add_scalars('R2', tag_scalar_dict={'test': R2_test, 'test_shift': R2_test_shift}, global_step=epoch)

        # better epoch save model, and save best performance model
        epoch_model_save_name = 'pka_net_epoch{}.pt'.format(epoch)
        best_model_save_name = 'pka_net_best.pt'
        if mini_loss > RMSE_test:
            model_save_path = os.path.join(model_save_dir, epoch_model_save_name)
            torch.save(net.state_dict(), model_save_path)

        # save best epoch model as best model
        if mini_loss > RMSE_test:
            model_save_path = os.path.join(model_save_dir, best_model_save_name)
            torch.save(net.state_dict(), model_save_path)
            mini_loss = RMSE_test
            best_model_name = epoch_model_save_name

        print(gc.collect())
    time2 = time.time()
    print('use time: {} min'.format((time2 - time1) / 60))
    print('best_model_name: {}'.format(best_model_name))
    print('model save dir: {}'.format(model_save_dir))
    print('train_info save dir: {}'.format(train_info_save_dir))
    print('Finished Training')


if __name__ == '__main__':
    train()
