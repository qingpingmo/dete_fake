import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger


from dingtalkchatbot.chatbot import DingtalkChatbot
robot_password = "SEC80db50fddb8d533cacb7584be8c917d5ab072cca186827d2d0a974eb8cfc8d82"
robot_webhook = "https://oapi.dingtalk.com/robot/send?access_token=50bd20736cf5a1aa988127193f20f82e0737e57637fa4a8a7e5ff65a2c91c512"
robot = DingtalkChatbot(robot_webhook, robot_password)

def send(msg):
    if isinstance(msg,list):
        msg = ';\n'.join([str(m) for m in msg])
    try:
        robot.send_text(str(msg))
    except:
        pass


vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataroot = os.path.join(opt.dataroot,'test')
    print(dataroot)
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    cmd = ' '.join(list(sys.argv) )
    print(cmd)
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    
    def testmodel(epoch=0):
        print('*'*25);accs = [];aps = [];logs=[f"Testing end of {epoch}"]
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(dataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.loadSize = 224
            Testopt.no_resize = False    # testing without resizing by default
            Testopt.no_crop = True    # testing without resizing by default
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc);aps.append(ap)
            logs.append("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
            print(logs[-1])
        logs.append("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
        print(logs[-1]);print('*'*25) 
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        if np.array(accs).mean()*100>86. : send([cmd, os.getcwd()]+logs)
        return round(np.array(accs).mean()*100, 4)
    model.eval();testmodel(0);
    model.train()

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()
            #print(model.model.conv1.weight.grad)
            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        testacc = testmodel(epoch)
        model.save_networks(str(epoch)+'_'+str(testacc))
        print(os.getcwd())
        print(cmd)
        print('saving the latest model %s (epoch %d, model.total_steps %d)' %(opt.name, epoch, model.total_steps))
        #testmodel()


        model.train()
    model.save_networks('last')
