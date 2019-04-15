import numpy as np
from dataset import to_dataloader
import torch
import copy
from args import device


def resample(Xlp, Xup, Xun, model_name=""):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name == "":
        raise ValueError("model name is unknown.")
    print("======start resampling======")
    n_Xup = Xup.shape[0]
    n_Xlp = Xlp.shape[0]
    print("unlabeled positive:", n_Xup)
    print("unlabeled negative:", Xun.shape[0])
    print("labeled positive:", n_Xlp)
    dim = Xup.size//len(Xup)

    # positive for 1, negative for 0
    Yup = np.ones(Xup.shape[0])
    Yun = np.zeros(Xun.shape[0])

    Xu = np.asarray(np.concatenate(
        (Xup, Xun), axis=0), dtype=np.float32)
    Yu_f = np.asarray(np.concatenate(
        (Yup, Yun)), dtype=np.float32).reshape((-1, 1))
    Yu = np.asarray(np.concatenate(
        (Yup, Yun)), dtype=np.int32)

    perm = np.random.permutation(len(Yu))
    Xu, Yu_f, Yu = Xu[perm], Yu_f[perm], Yu[perm]

    if model_name == "cnn":
        batch_size = 500
    else:
        batch_size = 30000
    unlabel_dataset = to_dataloader(
        Xu, Yu_f, batchsize=batch_size)  # create positive and negative datsets in unlabeled dataset
    unlabel_val_dataset = to_dataloader(
        Xu, Yu, batchsize=batch_size)  # create dev set

    print("======train logistic regression in unlabeled data======")
    # init model
    from train import select_model
    model = copy.deepcopy(select_model(model_name)(dim)).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=0.005)
    loss_func = torch.nn.BCEWithLogitsLoss()
    print(model)

    # train model with logistic regression
    for epoch in range(100):
        model.train()
        for data, target in unlabel_dataset:          # for each training step
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad()                # clear gradients for next train
            output = model(data)            # get output for every net
            loss = loss_func(output, target)  # compute loss for every net
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
        if (epoch+1) % 50 == 0:
            _, _, error_rate = model.error(
                unlabel_val_dataset, is_logistic=True)
            print("Epoch:{0}, error rate:{1}".format(epoch+1, error_rate))

    print("calculate p(y=+1|x)...")
    model.eval()

    print("resampling...")
    resample_from_unlable = True

    if resample_from_unlable:
        # resample from unlabeled positive data
        prob = torch.sigmoid(model(torch.from_numpy(
            Xup).to(device))).detach().cpu().numpy()

        prob = np.reshape(np.power(prob, 10), n_Xup)
        # normalize
        nor_prob = prob/np.sum(prob)
        # resample prob
        choose_index = np.random.choice(n_Xup, n_Xlp, replace=True, p=nor_prob)
        resample_Xlp = Xup[choose_index]
    else:
        prob = torch.sigmoid(model(torch.from_numpy(
            Xlp).to(device))).detach().cpu().numpy()

        prob = np.reshape(np.power(prob, 10), n_Xlp)
        # normalize
        nor_prob = prob/np.sum(prob)
        # resample prob
        choose_index = np.random.choice(n_Xlp, n_Xlp, replace=True, p=nor_prob)
        resample_Xlp = Xlp[choose_index]

    print("======finish resampling with positive data======")
    return resample_Xlp
