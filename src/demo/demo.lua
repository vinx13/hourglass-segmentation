require 'paths'
require 'image'
require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
nnlib = nn

paths.dofile('../models/hg.lua')
paths.dofile('predict.lua')
model = torch.load('../../exp/voc/default/model_55.t7')
model:evaluate()


function demo(imgname, bbox)
  print('--------------------')
  print('Processing '.. imgname)
  img = image.load(imgname)
  output = predict(img,bbox)
  img = image.drawRect(img, bbox[1], bbox[2], bbox[3], bbox[4]) -- draw bounding box
  output = torch.cat(img, output, 2)
  outname = imgname:sub(0,-5)..'_out.jpg'
  image.save(outname, output)
end

demo('2008_000652.jpg', torch.Tensor({1,7,238,210}))
demo('2008_000726.jpg', torch.Tensor({54,37,320,230}))
demo('2009_001103.jpg', torch.Tensor({137,71,354,331}))
