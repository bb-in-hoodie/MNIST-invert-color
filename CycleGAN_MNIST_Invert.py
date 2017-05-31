import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time

# ver 0.4
# discriminator : convolutional layers :
#	(1*28*28 -> 16*14*14 -> 32*7*7 -> 1)
# generator : convolutional layers & deconvolutional layers :
#	(1*28*28 -> 16*14*14 -> 16*14*14 -> 1*28*28)
# loss_dis : mean squared
# loss_gen : mean squared
# loss_cc : mean squared
# batch size : 100 (50 for black, 50 for white)
# epoch size : 300

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, stride=2, kernel_size=4, padding=1), # 28*28 -> 14*14
			nn.BatchNorm2d(16),
			nn.LeakyReLU()
		)
		self.layer2 = nn.Sequential(
		    nn.Conv2d(16, 32, stride=2, kernel_size=4, padding=1), # 14*14 -> 7*7
		    nn.BatchNorm2d(32),
		    nn.LeakyReLU()
		)
		self.fc = nn.Linear(7*7*32, 1)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.layer1 = nn.Sequential(
		    nn.Conv2d(1, 16, stride=2, kernel_size=4, padding=1), # 28*28 -> 14*14
		    nn.BatchNorm2d(16),
		    nn.LeakyReLU()
		)
		self.layer2 = nn.Sequential(
		    nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1), # 14*14 -> 14*14
		    nn.BatchNorm2d(16),
		    nn.LeakyReLU()
		)
		self.layer3 = nn.Sequential(
		    nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1), # 14*14 -> 28*28
		    nn.Tanh()
		)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		return out

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
dset = torchvision.datasets.MNIST(root = '/home/blackbindy/Documents/PytorchProjects/data/MNIST', transform = transforms, download = True)
dloader = torch.utils.data.DataLoader(dset, batch_size = 100, shuffle = True)

# Initial time
init_time = time.time()

# Discriminator and generator
dis_b = Discriminator()
gen_b = Generator()
dis_w = Discriminator()
gen_w = Generator()

def ZeroGrad():
	dis_b.zero_grad()
	gen_b.zero_grad()
	dis_w.zero_grad()
	gen_w.zero_grad()

# Convert the modules to the cuda modules if available
if(torch.cuda.is_available()):
	dis_b = dis_b.cuda()
	gen_b = gen_b.cuda()
	dis_w = dis_w.cuda()
	gen_w = gen_w.cuda()

optim_dis_w = optim.Adam(dis_w.parameters(), lr=0.0001)
optim_dis_b = optim.Adam(dis_b.parameters(), lr=0.0001)
optim_gen_w = optim.Adam(gen_w.parameters(), lr=0.0001)
optim_gen_b = optim.Adam(gen_b.parameters(), lr=0.0001)

print (dis_w)
print (gen_w)
print ()

for epoch in range(300):
	for i, batch in enumerate(dloader, 0):
		images = batch[0] # 100, 28, 28
		div_num = int(images.size(0)/2)

		# Split images into two groups
		raw_imgs_b = images[0:div_num]
		raw_imgs_w = images[div_num:images.size(0)] * -1 # Invert all the numbers

		imgs_b = Variable(raw_imgs_b)
		imgs_w = Variable(raw_imgs_w)

		num_b = imgs_b.size(0)
		num_w = imgs_w.size(0)

		ones_b = Variable(torch.ones(num_b))
		ones_w = Variable(torch.ones(num_w))

		# Convert the tensor to the cuda tensor if available
		if(torch.cuda.is_available()):
			imgs_b = imgs_b.cuda()
			imgs_w = imgs_w.cuda()
			ones_b = ones_b.cuda()
			ones_w = ones_w.cuda()

		### Training discriminator
		#1. real_loss_dis_b + real_loss_dis_w for real images
		ZeroGrad()

		out_b = dis_b(imgs_b)
		real_loss_dis_b = torch.mean((out_b - ones_b)**2)

		out_w = dis_w(imgs_w)
		real_loss_dis_w = torch.mean((out_w - ones_w)**2)

		real_loss = real_loss_dis_b + real_loss_dis_w
		real_loss.backward()

		optim_dis_b.step()
		optim_dis_w.step()

		#2. fake_loss_dis_b + fake_loss_dis_w for fake images
		ZeroGrad()

		fake_imgs_b = gen_b(imgs_w)
		out_b = dis_b(fake_imgs_b)
		fake_loss_dis_b = torch.mean(out_b**2) # num of zeros_w = num of fake_imgs_b = num of imgs_w

		fake_imgs_w = gen_w(imgs_b)
		out_w = dis_w(fake_imgs_w)
		fake_loss_dis_w = torch.mean(out_w**2) # num of zeros_b = num of fake_imgs_w = num of imgs_b

		fake_loss = fake_loss_dis_b + fake_loss_dis_w
		fake_loss.backward()

		optim_dis_b.step()
		optim_dis_w.step()

		### Training generator
		#1. loss_gen_w + loss_cc_b (black -> white -> black)
		ZeroGrad()

		fake_imgs_w = gen_w(imgs_b)
		out_w = dis_w(fake_imgs_w)
		loss_gen_w = torch.mean((out_w - ones_b)**2)

		recvd_imgs_b = gen_b(fake_imgs_w) # recvd : recovered
		loss_cc_b = torch.mean((imgs_b - recvd_imgs_b)**2)

		loss_bwb = loss_gen_w + loss_cc_b
		loss_bwb.backward()
		optim_gen_w.step()

		#2. loss_gen_b + loss_cc_w (white -> black -> white)
		ZeroGrad()

		fake_imgs_b = gen_b(imgs_w)
		out_b = dis_b(fake_imgs_b)
		loss_gen_b = torch.mean((out_b - ones_w)**2)

		recvd_imgs_w = gen_w(fake_imgs_b) # recvd : recovered
		loss_cc_w = torch.mean((imgs_w - recvd_imgs_w)**2)

		loss_wbw = loss_gen_b + loss_cc_w
		loss_wbw.backward()
		optim_gen_b.step()

		# Printing the loss
		if(i % 150 == 0):
			print("[%d, %d]-------------------------------------------"
				%(epoch, i))
			print("Discriminator - dis_b loss : %.4f, dis_w loss : %.4f"
				%((real_loss_dis_b.data[0] + fake_loss_dis_b.data[0]), (real_loss_dis_w.data[0] + fake_loss_dis_w.data[0])))
			print("Generator - gen_b loss : %.4f, gen_w loss : %.4f"
				%(loss_gen_b.data[0], loss_gen_w.data[0]))
			print("Cycle Consisteny - cc_b loss : %.4f, cc_w loss : %.4f"
				%(loss_cc_b.data[0], loss_cc_w.data[0]))

	# Saving real images and fake images as png image files
	torchvision.utils.save_image(raw_imgs_b, "./result/" + str(epoch) + "_BtoW_a.png")
	torchvision.utils.save_image(raw_imgs_w, "./result/" + str(epoch) + "_WtoB_a.png")

	fake_imgs_w = fake_imgs_w.view(fake_imgs_w.size(0), 1, 28, 28)
	torchvision.utils.save_image(fake_imgs_w.data, "./result/" + str(epoch) + "_BtoW_b.png")
	fake_imgs_b = fake_imgs_b.view(fake_imgs_b.size(0), 1, 28, 28)
	torchvision.utils.save_image(fake_imgs_b.data, "./result/" + str(epoch) + "_WtoB_b.png")

	# Printing the execution time
	exec_time = time.time() - init_time
	hours = int(exec_time/3600)
	mins = int((exec_time%3600)/60)
	secs = int((exec_time%60))
	print("\nExecution time : %dh %dm %ds"%(hours, mins, secs))
	print("====================================================\n")

torch.save(dis_w.state_dict(), './invert_dis_w.pkl')
torch.save(gen_w.state_dict(), './invert_gen_w.pkl')
torch.save(dis_b.state_dict(), './invert_dis_b.pkl')
torch.save(gen_b.state_dict(), './invert_gen_b.pkl')

# Execution time
exec_time = time.time() - init_time
hours = int(exec_time/3600)
mins = int((exec_time%3600)/60)
secs = int((exec_time%60))
print("====================================================")
print("Final execution time : %dh %dm %ds"%(hours, mins, secs))
print("====================================================")