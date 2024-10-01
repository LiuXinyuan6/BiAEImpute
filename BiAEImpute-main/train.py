import argparse

import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import RandomSampler,BatchSampler
# from tensorboardX import SummaryWriter
from utils import *
from datasets import *
from model import *
from loss import *
import itertools

config = Config()


if __name__ == '__main__':
    # define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp_id',default=1.0,required=False,help='experiment id')
    argparser.add_argument('--datasets',default='Zeisel.csv',required=True,help='dataset')
    argparser.add_argument('--mask_ratio', default=0.4, type=float ,required=True,help='if 0:no dropout else dropout mask_ratio% with the ground-truth data')
    argparser.add_argument('--normalization', type=bool, default=True, required=True,help='if False:no normalization else normalization')
    argparser.add_argument('--latent_size', default=128, type=int, required=False, help='latent space size')
    argparser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    argparser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    argparser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    argparser.add_argument('--eps',default=500,type=int,help='training epochs')
    argparser.add_argument('--bs_row',default=31,type=int,help='batch_size for row-wise autoencoder')
    argparser.add_argument('--bs_col', default=200, type=int, help='batch_size for col-wise autoencoder')
    argparser.add_argument('--cell_nm', default=3005, type=int, help='cell numbers')
    argparser.add_argument('--gene_nm', default=19972, type=int, required=False, help='gene numbers')
    argparser.add_argument('--gpu_id',default='0',help='which gpu to use')
    args = argparser.parse_args()

    # initializing dataset
    dataset = SingleCell(
        config.data_root,
        args.datasets,
    )

    # tensorboard
    #writer = SummaryWriter(log_dir=config.data_root+'logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load processed scRNA-seq dataset
    sc_dataset = dataset.load_data(mask_ratio=args.mask_ratio, normalization=args.normalization).T # 行细胞 列基因
    scdata = torch.Tensor(sc_dataset)
    if torch.cuda.is_available():
        scdata = scdata.to(device)

    # dataloader
    dataset_row = RowDataset(scdata)
    data_loader_row = DataLoader(dataset_row, batch_size=args.bs_row, shuffle=True, drop_last=False)
    dataset_col = ColDataset(scdata)
    data_loader_col = DataLoader(dataset_col, batch_size=args.bs_col, shuffle=True, drop_last=False)

    # model
    row_encoder = Row_Encoder(args.gene_nm, args.latent_size)
    row_decoder = Row_Decoder(args.latent_size,args.gene_nm)
    col_encoder = Col_Encoder(args.cell_nm, args.latent_size) #
    col_decoder = Col_Decoder(args.latent_size,args.cell_nm)

    # loss
    # RegularizeLoss = SquareRegularizeLoss(p=config.p)
    # if torch.cuda.is_available():
    #     RegularizeLoss = RegularizeLoss.to(device)

    if torch.cuda.is_available():
        row_encoder.to(device)
        row_decoder.to(device)
        col_encoder.to(device)
        col_decoder.to(device)

    # optimizer
    optimizer = torch.optim.Adam(
        itertools.chain(row_encoder.parameters(), row_decoder.parameters(), col_encoder.parameters(), col_decoder.parameters()), lr=args.lr, betas=(args.b1, args.b2)
    )


    for epoch in range(args.eps):
        epoch_reconstruct_loss  = 0.0
        iter_num = 0
        for (row_data, row_idx), (col_data, col_idx) in zip(data_loader_row, data_loader_col):
            iter_num = iter_num+1
            if torch.cuda.is_available():
                row_data = row_data.to(device)
                col_data = col_data.to(device)
            # forward
            row_latent = row_encoder(row_data)
            row_output = row_decoder(row_latent)
            col_latent = col_encoder(col_data)
            col_output = col_decoder(col_latent)
            # mask matrix
            mask_row = torch.where(row_data == 0, torch.zeros_like(row_data), torch.ones_like(row_data))
            mask_col = torch.where(col_data == 0, torch.zeros_like(col_data), torch.ones_like(col_data))

            # calculating loss respectively
            loss_row = ((row_output.mul(mask_row)-row_data)**2).sum()/mask_row.sum()
            loss_col = ((col_output.mul(mask_col)-col_data)**2).sum()/mask_col.sum()
            row_cross_points = row_output[:, col_idx]
            col_cross_points = col_output[:, row_idx].T
            loss_cross = ((row_cross_points - col_cross_points) ** 2).mean()
            # total loss
            total_loss = loss_row + loss_col + loss_cross

            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_reconstruct_loss += total_loss

        print("第{}次reconstruct_loss:".format(epoch), epoch_reconstruct_loss)

        # record the loss in each iteration
        #writer.add_scalar('reconstruct_loss', epoch_reconstruct_loss, epoch)

    print("训练结束")
    # close summaryWriter after training
    #writer.close()

    # save model after training
    torch.save(row_encoder.state_dict(),config.data_root+"row_encoder.pth")
    torch.save(row_decoder.state_dict(), config.data_root + "row_decoder.pth")
    torch.save(col_encoder.state_dict(),config.data_root+"col_encoder.pth")
    torch.save(col_decoder.state_dict(), config.data_root + "col_decoder.pth")
