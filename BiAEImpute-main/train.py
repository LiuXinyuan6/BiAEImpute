import argparse

import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import RandomSampler,BatchSampler
from tensorboardX import SummaryWriter
from utils import *
from datasets import *
from model import *
from loss import *
import itertools



config = Config()


# best_loss = 1e10
if __name__ == '__main__':
    #define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp_id',default=1.0,required=False,help='experiment id')
    argparser.add_argument('--datasets',default='Klein_normalization_mask40.csv',required=False,help='dataset_name')
    argparser.add_argument('--mask_ratio', default=0, required=False,help='if 0:no dropout else dropout mask_ratio% with the ground-truth data')
    #argparser.add_argument('--normalization', default=True, required=False,help='if True:need normalization else not need')
    argparser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    argparser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")#0.5
    argparser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    argparser.add_argument('--eps',default=500,type=int,help='reset training epochs of training')
    argparser.add_argument('--bs_row',default=272,type=int,help='batch_size for training only')
    argparser.add_argument('--bs_col', default=242, type=int, help='batch_size for training only')
    argparser.add_argument('--cell_nm', default=2717, type=int, help='batch_size for training only')
    argparser.add_argument('--gpu_id',default='0',help='which gpu to use')
    argparser.add_argument('--gene_nm', default=24175, type=int, required=False, help='which gpu to use')
    argparser.add_argument('--latent_size', default=128, type=int, required=False, help='which gpu to use')
    args = argparser.parse_args()

    dataset = SingleCell(
        config.data_root,
        args.datasets,
    )

    #tensorboard
    #writer = SummaryWriter(log_dir=config.data_root+'logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load processed scRNA-seq dataset
    sc_dataset = dataset.load_data(mask_ratio=args.mask_ratio).T # 行细胞 列基因#
    scdata = torch.Tensor(sc_dataset)
    if torch.cuda.is_available():
        scdata = scdata.to(device)


    
    dataset_row = RowDataset(scdata)
    data_loader_row = DataLoader(dataset_row, batch_size=args.bs_row, shuffle=True,drop_last=False)
    dataset_col = ColDataset(scdata)
    data_loader_col = DataLoader(dataset_col, batch_size=args.bs_col, shuffle=True,drop_last=False)
    # data_iter = iter(data_loader)
    # sample_batch = next(data_iter)



    #model
    row_encoder = Row_Encoder(args.gene_nm, args.latent_size)
    row_decoder = Row_Decoder(args.latent_size,args.gene_nm)
    col_encoder = Col_Encoder(args.cell_nm, args.latent_size) #
    col_decoder = Col_Decoder(args.latent_size,args.cell_nm)

    #Loss
    # reconstruct_loss = nn.L1Loss()
    RegularizeLoss = SquareRegularizeLoss(p=config.p)
    if torch.cuda.is_available():
        RegularizeLoss = RegularizeLoss.to(device)

    if torch.cuda.is_available():
        row_encoder.to(device)
        row_decoder.to(device)
        col_encoder.to(device)
        col_decoder.to(device)

    #optimizer
    optimizer = torch.optim.Adam(
        itertools.chain(row_encoder.parameters(), row_decoder.parameters(), col_encoder.parameters(), col_decoder.parameters()), lr=args.lr, betas=(args.b1, args.b2)
    )


    for epoch in range(args.eps):
        epoch_reconstruct_loss  = 0.0
        iter_num = 0
        for (row_data, row_idx), (col_data, col_idx) in zip(data_loader_row, data_loader_col):
        #for row_idx, col_idx in data_loader:#I J
            iter_num = iter_num+1
            # row_data = scdata[row_idx, :]
            # col_data = scdata[:, col_idx]
            if torch.cuda.is_available():
                row_data = row_data.to(device)
                col_data = col_data.to(device)

            row_latent = row_encoder(row_data)
            row_output = row_decoder(row_latent)
            col_latent = col_encoder(col_data)
            col_output = col_decoder(col_latent)


            mask_row = torch.where(row_data == 0, torch.zeros_like(row_data), torch.ones_like(row_data))
            mask_col = torch.where(col_data == 0, torch.zeros_like(col_data), torch.ones_like(col_data))


            loss_row = ((row_output.mul(mask_row)-row_data)**2).sum()/mask_row.sum()
            loss_col = ((col_output.mul(mask_col)-col_data)**2).sum()/mask_col.sum()

            loss_regularize = RegularizeLoss(row_latent) + RegularizeLoss(col_latent)


            row_cross_points = row_output[:, col_idx]
            col_cross_points = col_output[:, row_idx].T#.transpose(0,1)
            loss_cross = ((row_cross_points - col_cross_points) ** 2).mean()


            total_loss = loss_row + loss_col + loss_cross


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_reconstruct_loss += total_loss



        print("第{}次reconstruct_loss:".format(epoch), epoch_reconstruct_loss)
        #dataset.on_epoch_end()

        #writer.add_scalar('reconstruct_loss', epoch_reconstruct_loss, epoch)

    print("训练结束")
    #writer.close()
    torch.save(row_encoder.state_dict(),config.data_root+"row_encoder.pth")
    torch.save(row_decoder.state_dict(), config.data_root + "row_decoder.pth")
    torch.save(col_encoder.state_dict(),config.data_root+"col_encoder.pth")
    torch.save(col_decoder.state_dict(), config.data_root + "col_decoder.pth")










