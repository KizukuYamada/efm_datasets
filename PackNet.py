import torch
import cv2 
import pdb


packnet_model = torch.hub.load("TRI-ML/vidar", "PackNet", pretrained=True, trust_repo=True)
rgb = torch.tensor(cv2.imread('examples/ddad_sample.png')).permute(2,0,1).unsqueeze(0)/255.

depth_pred = packnet_model(rgb)
pdb.set_trace()