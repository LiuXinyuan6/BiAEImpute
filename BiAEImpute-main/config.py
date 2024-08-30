

class Config(object):
    # data directory root
    data_root = 'D:\\BiAEImpute\\data\\'

    # model configs
    #middle_layer_size = [512, 256, 128, 256, 512]
    # middle_layer_size = [64, 32, 16, 32, 64]
    #middle_layer_size = [512, 256, 512]
    #middle_layer_size = [64, 32, 16, 32, 64]

    # regularized loss, (1-(x1^2+...+xn^2))^p
    p = 2

    # format for saving encoded results
    #formt = 'txt'   

    latent_size = 128