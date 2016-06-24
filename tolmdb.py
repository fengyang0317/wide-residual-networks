import lmdb
import numpy as np
import h5py
import caffe

f = h5py.File('cifar.h5', 'r')
train_data = f['trainData']['data'][()]
train_label = f['trainData']['labels'][()] - 1

test_data = f['testData']['data'][()]
test_label = f['testData']['labels'][()] - 1

def writelmdb(lmdb_name, data, label):
    label = label.astype(int)
    db = lmdb.open(lmdb_name, map_size=int(1e12))
    txn = db.begin(write=True)
    for i in xrange(label.size):
        str_id = '{:08}'.format(i)
        datum = caffe.io.array_to_datum(data[i], label[i])
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        if (i + 1) % 1000 == 0:
            txn.commit()
            print lmdb_name, i + 1
            txn = db.begin(write=True)
    txn.commit()
    db.close()

ix = np.random.permutation(train_label.size)
train_data = train_data[ix]
train_label = train_label[ix]


t_data = np.empty((train_label.size, 3, 40, 40), np.float64)
t_data[:, :, 4:36, 4:36] = train_data
t_data[:, :, :4, :4] = train_data[:, :, 4:0:-1, 4:0:-1]
t_data[:, :, :4, -4:] = train_data[:, :, 4:0:-1, -2:-6:-1]
t_data[:, :, -4:, :4] = train_data[:, :, -2:-6:-1, 4:0:-1]
t_data[:, :, -4:, -4:] = train_data[:, :, -2:-6:-1, -2:-6:-1]

t_data[:, :, :4, 4:36] = train_data[:, :, 4:0:-1, :]
t_data[:, :, -4:, 4:36] = train_data[:, :, -2:-6:-1, :]
t_data[:, :, 4:36, :4] = train_data[:, :, :, 4:0:-1]
t_data[:, :, 4:36, -4:] = train_data[:, :, :, -2:-6:-1]


writelmdb('wcifar_train', t_data, train_label)
writelmdb('wcifar_test', test_data, test_label)