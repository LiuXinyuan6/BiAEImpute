import argparse
from tqdm import tqdm
from config import Config
from utils import *
from datasets import *
from model import *
from loss import *

config = Config()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu_id', default='0', help='which gpu to use')
    argparser.add_argument('--gene_nm', default=24175, type=int, required=False, help='which gpu to use')
    argparser.add_argument('--cell_nm', default=2717, type=int, help='batch_size for training only')
    argparser.add_argument('--latent_size', default=128, type=int, required=False, help='which gpu to use')

    args = argparser.parse_args()

    X = pd.read_csv(config.data_root+'Klein_normalization_mask40.csv', sep=',', index_col=0)#header=None, index_col=None
    X = X.T  #3000行cell*1477列gene
    X = X.values

    row_encoder = Row_Encoder(args.gene_nm, args.latent_size)
    row_decoder = Row_Decoder(args.latent_size,args.gene_nm)
    col_encoder = Col_Encoder(args.cell_nm, args.latent_size) #
    col_decoder = Col_Decoder(args.latent_size,args.cell_nm)

    load_row_encoder_model_path = config.data_root+"row_encoder.pth"
    load_row_decoder_model_path = config.data_root + "row_decoder.pth"
    load_col_encoder_model_path = config.data_root+"col_encoder.pth"
    load_col_decoder_model_path = config.data_root + "col_decoder.pth"
    row_encoder.load_state_dict(torch.load(load_row_encoder_model_path))#,map_location='cpu'
    row_decoder.load_state_dict(torch.load(load_row_decoder_model_path))
    col_encoder.load_state_dict(torch.load(load_col_encoder_model_path))
    col_decoder.load_state_dict(torch.load(load_col_decoder_model_path))

    row_encoder.eval()
    row_decoder.eval()
    col_encoder.eval()
    col_decoder.eval()

    if torch.cuda.is_available():
        row_encoder.to('cuda')
        row_decoder.to('cuda')
        col_encoder.to('cuda')
        col_decoder.to('cuda')

    y_dec_row = []
    with torch.no_grad():
        for cell in tqdm(X):
            # cell = torch.Tensor(cell).view(1, -1).to('cuda')
            cell = torch.Tensor(cell).view(1, -1)#1,2000
            if torch.cuda.is_available():
              cell = cell.to('cuda')
            print(cell.shape)
            row_enc = row_encoder(cell)
            row_dec = row_decoder(row_enc)
            # dec = dec.cpu().numpy().squeeze()
            row_dec = row_dec.cpu().numpy().squeeze()
            y_dec_row.append(row_dec)

    y_dec_row = np.array(y_dec_row)
    y_dec_row = y_dec_row.T
    np.savetxt(config.data_root + "outcome_row.csv", y_dec_row, delimiter=',')


    y_dec_col = []
    with torch.no_grad():
        for gene in tqdm(X.T):
            # cell = torch.Tensor(cell).view(1, -1).to('cuda')
            gene = torch.Tensor(gene).view(1, -1)#1,2000
            if torch.cuda.is_available():
              gene = gene.to('cuda')
            print(gene.shape)
            col_enc = col_encoder(gene)
            col_dec = col_decoder(col_enc)
            # dec = dec.cpu().numpy().squeeze()
            col_dec = col_dec.cpu().numpy().squeeze()
            y_dec_col.append(col_dec)

    y_dec_col = np.array(y_dec_col)
    y_dec_col = y_dec_col.T.T #1477*3000
    np.savetxt(config.data_root + "outcome_col.csv", y_dec_col, delimiter=',')

    y_dec_average = (y_dec_col + y_dec_row) / 2
    
    np.savetxt(config.data_root + "y_dec_average.csv", y_dec_average, delimiter=',')

    
    y_dec_pd = pd.read_csv(config.data_root+"y_dec_average.csv", sep=',', header=None, index_col=False)
    y_dec_pd = y_dec_pd.values
    #identify_Data = pd.read_csv(config.data_root+"log_Zeisel_identify.csv", sep=',', header=None, index_col=False)
    identify_Data = pd.read_csv(config.data_root + "Klein_normalization_mask40.csv", sep=',', index_col=0)
    #identify_Data = identify_Data.T
    identify_Data = identify_Data.values
    save_Identify_trueValue(identify_Data,y_dec_pd)


