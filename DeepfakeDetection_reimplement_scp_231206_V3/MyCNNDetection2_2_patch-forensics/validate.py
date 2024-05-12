import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader

softmax = torch.nn.Softmax(dim = 1)

def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            # print()
            # tmp=img.numpy(); print(f'img shape:{tmp.shape}, max: {tmp.max()}, min: {tmp.min()}')
            in_tens = model( img.cuda() )
            # tmp=in_tens.cpu().numpy(); print(f'out shape:{tmp.shape}, max: {tmp.max()}, min: {tmp.min()}')
            n, c, h, w = in_tens.shape
            # tmp = in_tens; print(f' shape: {tmp.shape}  max: {tmp.max()}  min: {tmp.min()} ')
            # votes = torch.argmax(in_tens, dim=1).view(n, -1)
            # vote_predictions = torch.mean(  votes.float(), axis=1  )

            before_softmax_predictions =  softmax( torch.mean(in_tens, dim=(-1, -2)) )
            # tmp = before_softmax_predictions; print(f' shape: {tmp.shape}  max: {tmp.max()}  min: {tmp.min()} ')
            before_softmax_predictions =  before_softmax_predictions[:,-1].cpu().numpy()
            # tmp = before_softmax_predictions; print(f' shape: {tmp.shape}  max: {tmp.max()}  min: {tmp.min()} ')
            # after_softmax_predictions = torch.mean( softmax(in_tens), dim=(-1, -2) )

            # vote_predictions = torch.stack([1-vote_predictions, vote_predictions], axis=1).cpu().numpy()
            
            # labels  = label.reshape((-1,1,1)).expand(n, h, w)
            # avg_preds = torch.argmax( softmax(in_tens).mean(dim=(2,3)), dim=1 )
            # acc_D_avg = torch.mean( torch.eq( labels, avg_preds ).float())
            # exit()
            y_pred.extend(before_softmax_predictions.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
