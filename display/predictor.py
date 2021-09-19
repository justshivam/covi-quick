import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Normalize

def res(dataloders, model):
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloders['valid']):
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)
            print(outputs)
            _, preds = torch.max(outputs.data, 1)
            return preds.numpy()
        
def predict(MODEL_PATH='./display/resnet50.pth', DIR='./media'):

    data_dir = DIR
    stats = ((0.6414, 0.6414, 0.6414), (0.2608, 0.2608, 0.2608))
    transform1 = transforms.Compose([Resize((244,244)),ToTensor(), Normalize(*stats,inplace=True)])

    random_seed = 42
    torch.manual_seed(random_seed)
    dataset = ImageFolder(data_dir, transform=transform1)
    val_dl = DataLoader(dataset, len(dataset), num_workers=4, pin_memory=True)
    use_gpu = torch.cuda.is_available()
    resnet = models.resnet50(pretrained=True)
    inputs, labels = next(iter(val_dl))
    if use_gpu:
        resnet = resnet.cuda()
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())   
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = resnet(inputs)
    outputs.size()
    dloaders = {'train':val_dl, 'valid':val_dl}
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    # new final layer with 2 classes
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, 2)
    resnet.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    return res(dloaders, resnet)

if __name__ == '__main__':
    print(predict(MODEL_PATH='./resnet50.pth', DIR='../media'))