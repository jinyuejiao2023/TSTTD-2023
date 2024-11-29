from torch.utils.data import DataLoader
from dataset.HTD_dataset import HTD_dataset
from evaluation import *
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spo
torch.set_default_tensor_type(torch.cuda.FloatTensor)

img_list = ['Muufl Gulfport','San Diego']
for i in range(len(img_list)):
    img_name = img_list[i]
    # dataset
    train_dataset = HTD_dataset('./data', img_name, target_abu=0)
    b = np.sum(train_dataset.groundtruth == 1)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))
    data_num = train_dataset.train_img.shape[0]
    # initialization
    band_num = train_dataset.test_img.shape[1]
    detector = torch.load("./model/{}.pth".format(img_name))
    with torch.no_grad():
        detector.eval()
        prior = train_dataloader.dataset.prior
        prior = torch.Tensor(prior)#(193,)
        test_img = torch.from_numpy(train_dataset.test_img.astype('float32')).cuda()  # (40000,189)
        result = np.zeros(data_num)
        features = np.zeros((data_num,32))
        feature_prior = np.zeros((1, 32))
        a = int(data_num/100)
        for j in range(a):
            test_img_p = test_img[100 * j:100 * (j + 1), :]
            prior_img_p = prior.unsqueeze(0)
            corrs, class_head = detector([prior_img_p, test_img_p])
            corrs = corrs.cpu()
            result[100 * j:100 * (j + 1)] = corrs[0:100]
    #---------------------------------------------------
    col, row = train_dataset.groundtruth.shape
    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
    print('--------------------------------------')
    auc_return, ft_return = plot_ROC(train_dataset.groundtruth.reshape(-1, 1, order='F'), [result], [img_name], img_name, show=False)

    spo.savemat('./scores_{}.mat'.format(img_name,auc_return[0]),{'score': result.reshape(col, row, order='F')})
    plt.imsave('./{}_viridis.eps'.format(img_name, auc_return[0]), result.reshape(col, row, order='F'), cmap='viridis')

