import caffe
from caffe import layers as L
from caffe import params as P

c1_kwargs = {
    'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
    'weight_filler': dict(type='xavier'),
    'bias_filler': dict(type='constant', value=0.0)
}
bn_kwargs = {
    #'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
    'eps': 0.001,
}


def wide_basic(net, from_layer, prefix, nInputPlane, nOutputPlane, stride):
    #assert from_layer in net.keys()

    bn_branch2a_name = '{}_bn_branch2a'.format(prefix)
    net[bn_branch2a_name] = L.BatchNorm(net[from_layer], in_place=True, **bn_kwargs)
    relu_branch2a_name = '{}_relu_branch2a'.format(prefix)
    net[relu_branch2a_name] = L.ReLU(net[bn_branch2a_name], in_place=True)
    branch2a_name = '{}_branch2a'.format(prefix)
    net[branch2a_name] = L.Convolution(net[relu_branch2a_name], num_output=nOutputPlane, kernel_size=3, pad=1,
                                       stride=stride, **c1_kwargs)

    bn_branch2b_name = '{}_bn_branch2b'.format(prefix)
    net[bn_branch2b_name] = L.BatchNorm(net[branch2a_name], in_place=True, **bn_kwargs)
    relu_branch2b_name = '{}_relu_branch2b'.format(prefix)
    net[relu_branch2b_name] = L.ReLU(net[bn_branch2b_name], in_place=True)
    branch2b_name = '{}_branch2b'.format(prefix)
    net[branch2b_name] = L.Convolution(net[relu_branch2b_name], num_output=nOutputPlane, kernel_size=3, pad=1,
                                       stride=1, **c1_kwargs)
    if nInputPlane != nOutputPlane:
        branch1_name = '{}_branch1'.format(prefix)
        net[branch1_name] = L.Convolution(net[relu_branch2a_name], num_output=nOutputPlane, kernel_size=1,
                                          stride=stride, pad=0, **c1_kwargs)
        net[prefix] = L.Eltwise(net[branch2b_name], net[branch1_name])
    else:
        net[prefix] = L.Eltwise(net[branch2b_name], net[from_layer])


def layer(net, from_layer, prefix, nInputPlane, nOutputPlane, count, stride):
    l_name = '{}_0'.format(prefix)
    wide_basic(net, from_layer, l_name, nInputPlane, nOutputPlane, stride)
    for i in xrange(1, count):
        c_name = '{}_{}'.format(prefix, i)
        wide_basic(net, l_name, c_name, nInputPlane, nOutputPlane, 1)
        l_name = c_name


def wide_resnet(lmdb, batch_size, num_class, depth = 10, widen_factor = 1, transform_param=dict()):
    assert (depth - 4) % 6 ==0, '%6=0'
    n = (depth - 4) / 6
    nStages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    net = caffe.NetSpec()
    net.data, net.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             ntop=2, transform_param=transform_param)
    net['conv1'] = L.Convolution(net['data'], num_output=nStages[0], kernel_size=3, stride=1, pad=1, **c1_kwargs)
    layer(net, 'conv1', 'res2a', nStages[0], nStages[1], n, 1)
    layer(net, 'res2a_{}'.format(n-1), 'res3a', nStages[1], nStages[2], n, 2)
    layer(net, 'res3a_{}'.format(n-1), 'res4a', nStages[2], nStages[3], n, 2)
    net['bn'] = L.BatchNorm(net['res4a_{}'.format(n-1)], in_place=True, **bn_kwargs)
    net['relu'] = L.ReLU(net['bn'], in_place=True)
    #net['pool'] = L.Pooling(net['relu'], pool=P.Pooling.AVE, global_pooling=True)
    net['pool'] = L.Pooling(net['relu'], pool=P.Pooling.AVE, kernel_size=8)
    net['fc'] = L.InnerProduct(net['pool'], num_output=num_class, weight_filler=dict(type='xavier'),
                               bias_filler=dict(type='constant', value=0.0))
    net['accuracy'] = L.Accuracy(net['fc'], net.label)
    net['loss'] = L.SoftmaxWithLoss(net['fc'], net.label)

    return net.to_proto()

with open('auto_train.prototxt', 'w') as f:
    train_transform_param = {
        'mirror': True,
        'crop_size': 32
    }
    f.write('name: "wide resnet"\n')
    f.write(str(wide_resnet('wcifar_train', 100, 10, transform_param=train_transform_param)))

with open('auto_test.prototxt', 'w') as f:
    f.write('name: "wide resnet"\n')
    f.write(str(wide_resnet('wcifar_test', 100, 10)))