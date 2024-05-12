from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = '../test/'

# list of synthesis algorithms
# vals = ['progan', 'proganc2', 'stylegan', 'styleganc2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal', 'proganc2', 'styleganc2']
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake', 'proganc2', 'styleganc2']
# multiclass = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]

vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

# vals = ['proganc2', 'styleganc2']
# multiclass = [0, 0]

# model
# model_path = 'weights/resnet50_20220517backup_state_dict.pth'
# model_path = 'checkpoints/resnet_noaug/model_epoch_latest.pth'
