from torch import float32
import torch.optim as optim
from torch.utils.data import DataLoader
import HTD_dataset
from evaluation import *
from vit import ViT
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(torch.cuda.is_available())
lr = 0.0001
batch_size = 64
epoch_num = 50

num_layers = 6
embedding_dim = 32
patch_dim = 5  # group_wise
num_heads = 2
attention_mlp_hidden = 32
dropout_rate = 0.1
emb_dropout = 0.1

img_list = ['Muufl Gulfport','San Diego']

for i in range(len(img_list)):
    img_name = img_list[i]
    # dataset
    train_dataset = HTD_dataset('./data', img_name, target_abu=0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator(device='cuda'))
    data_num = train_dataset.train_img.shape[0]
    # initialization
    band_num, endmember_num = train_dataset.test_img.shape[1], train_dataset.vd_num
    num_patches = band_num
    # model build
    detector = ViT(
        num_layers,
        num_patches,
        patch_dim,
        embedding_dim,
        num_heads,
        dropout_rate,
        attention_mlp_hidden,
        emb_dropout,
    ).cuda()
    op = optim.Adam(detector.parameters(), lr=lr, weight_decay=0.0)
    # model training
    detector.train()
    best_result = 0
    for e in range(epoch_num):
        for _ in range(round(data_num / batch_size) + 1):
            prior, back = next(iter(train_dataloader))
            prior_s = prior[0].unsqueeze(0)# (1, 189)
            a = prior_s.norm(2, dim=1, keepdim=True)
            back_ = back * prior.norm(2, dim=1, keepdim=True) / back.norm(2, dim=1, keepdim=True).to(prior.device)
            abundance = 0.1 * torch.rand(prior.shape[0])[:, None].to(prior.device)
            pseudo_targets = ((1 - abundance) * prior + abundance * back_).to(prior.device)  # mixed_target +
            prior_branch = prior_s.cuda()  # prior prior
            siamese_branch = torch.cat([back, pseudo_targets]).cuda()  # -    +
            label = torch.cat([torch.zeros(prior.shape[0]), 1.0 * torch.ones(prior.shape[0])]).to(siamese_branch.device)
            # --------------------输入网络-----------------------
            corrs, class_head = detector([prior_branch, siamese_branch])
            # ---------------------求loss-----------------------
            loss = detector.loss(label, corrs, class_head)
            op.zero_grad()
            loss.backward()
            op.step()
        print("epoch:{}, loss={}".format(e, loss))

        with torch.no_grad():
            detector.eval()
            prior = train_dataloader.dataset.prior
            prior = torch.Tensor(prior)
            data_num = train_dataset.test_img.shape[0]
            test_img = torch.from_numpy(train_dataset.test_img.astype('float32')).cuda()
            result = np.zeros(data_num)
            img_test = np.zeros((data_num, embedding_dim))
            pri_test = np.zeros((1, embedding_dim))

            a = int(data_num/100)
            for j in range(a):
                test_img_p = test_img[100 * j:100 * (j + 1), :]
                prior_img_p = prior.unsqueeze(0)
                corrs, class_head = detector([prior_img_p, test_img_p])
                corrs = corrs.cpu()
                # -------------------------------------------------------------
                result[100 * j:100 * (j + 1)] = corrs[0:100]
        #---------------------------------------------------
        col, row = train_dataset.groundtruth.shape
        auc_return, ft_return = plot_ROC(train_dataset.groundtruth.reshape(-1, 1, order='F'), [result], ['SFCTD'], img_name, show=False)
        print('epoch{}-{}-AUC:{}'.format(e, img_name, auc_return[0]))
        plt.imsave('./result/{}_{}.jpg'.format(i, img_name), result.reshape(col, row, order='F'),
                   cmap='gray', dpi=300)
        plt.imsave('./result/{}_{}_viridis_SFCTD.eps'.format(i,img_name),
                    result.reshape(col, row, order='F'), cmap='viridis')
        torch.save(detector, "./result/{}_{}.pth".format(i, img_name))
        if auc_return[0] > best_result:
            best_result = auc_return[0]
            torch.save(detector, "./result/{}_{}_best.pth".format(i, img_name))
            plt.imsave('./result/{}_{}_best.jpg'.format(i, img_name), result.reshape(col, row, order='F'), cmap='gray', dpi=300)
            plt.imsave('./result/{}_{}_viridis_best.eps'.format(i, img_name), result.reshape(col, row, order='F'),
                       cmap='viridis', dpi=300)
    print("--------------------------------------------------------")
    print('{}_{}-best_AUC:{}'.format(i, img_name, best_result))
    print("--------------------------------------------------------")
print("finished")