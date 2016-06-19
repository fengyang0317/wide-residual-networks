require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.imgproc'
local repl = require 'trepl'
local c = require 'trepl.colorize'
local json = require 'cjson'
paths.dofile'augmentation.lua'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

net = torch.load('logs/wide-resnet_tmp/model.t7')
net:evaluate()

local im = cv.imread {'2012-12-12_12_30_08.jpg'}

im = cv.resize {im, {640, 360}}
cv.imshow {'im', im}

im = im:permute(3, 1, 2):type('torch.FloatTensor')

im[{1,{},{}}]:add(-161.07047776)
im[{2,{},{}}]:add(-166.17654151)
im[{3,{},{}}]:add(-166.76638209)

im:div(255)

im = im:view(1, 3, im:size(2), im:size(3))

--im = torch.randn(1, 3, 64, 32)

print(im:size())
net:add(nn.View(2, 75, 153):setNumInputDims(1):cuda())
--net:add(nn.SoftMax():cuda())


output = net:forward(im:cuda())

--[[
cudnn.convert(net, nn)
net = net:float()
st = os.clock()
output = net:forward(im)
print(os.clock() - st .. " second")
--]]

print(output:size())
--print(net:listModules())

output = output:double()
--output = nn.SoftMax():forward(output)
--output = output:view(2, 75, 153)
--output = output:float()
--output = output:permute(2, 3, 1)

m1 = output[{1,1,{},{}}]
m2 = output[{1,2,{},{}}]

--[[
m1 = torch.log(m1 - m1:min())
m2 = torch.log(m2 - m2:min())
m1 = m1 / m1:max()
m2 = m2 / m2:max()
--]]

m1 = cv.resize {m1, {640, 360}}
m2 = cv.resize {m2, {640, 360}}

cv.imshow {'a', m1}
cv.imshow {'b', m2}
cv.waitKey {0}
cv.imwrite {'ht.png', m2 * 255}