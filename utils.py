import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models
import torchvision.transforms as T 
from PIL import Image

DEVICE = torch.device('cpu')

class Net(nn.Module):
	def __init__(self, n_classes, pretrained = True, model ='resnext'):
		super(Net, self).__init__()
		if model == 'resnext':
			self.net = models.resnext50_32x4d(pretrained=pretrained)
			self.net.fc = nn.Linear(2048, n_classes)
		elif model == 'vgg19':
			self.net = models.vgg19(pretrained=pretrained)
			self.net.classifier[6] = nn.Linear(4096, n_classes)
	
	def forward(self, x):
		x = self.net(x)
		return x

def ImageTestTransform(image):
	transforms = T.Compose([
		T.Resize((224, 224)),
		T.ToTensor(),
		T.Normalize((0.4766, 0.4524, 0.3928),
				   (0.2272, 0.2225, 0.2208))
	])
	
	image_tensor = transforms(image)
	image_tensor = image_tensor.unsqueeze(0)
	return image_tensor

def predict(image_path, model=None, mapping=None):
	if model is not None:
		model = model.to(DEVICE)
	image = Image.open(image_path)
	image_tensor = ImageTestTransform(image)
	out = model(image_tensor.to(device=DEVICE, dtype=torch.float))
	out = out.squeeze(0)
	prob = F.softmax(out, 0)
	prob, pred = torch.max(prob, dim=0)
	pred = pred.detach().cpu().numpy()
	name = mapping[int(pred)]

	return {
	'out':out,
	'probability':prob,
	'prediction':pred,
	'dog':name
	}
