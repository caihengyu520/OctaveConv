# 卷积网络的训练数据为MNIST(28*28灰度单色图像)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from _load_cifar_train import load_cifar_train
from _load_cifar_test import load_cifar_test
from _get_num_params import get_num_params
# from _load_mnist_test_false import load_mnist_test_false
from _next_batch_ import next_batch_
import pandas as pd
import scipy.misc
# from _heatmap import heatmap

random.seed(200)

train_epochs = 1000    # 训练轮数
batch_update=1
batch_size   = 100     # 随机出去数据大小
display_step = 10     # 显示训练结果的间隔
learning_rate= 0.0001  # 学习效率
drop_prob    = 0.5     # 正则化,丢弃比例
fch_nodes    = 2048     # 全连接隐藏层神经元的个数
start=0

# 网络模型需要的一些辅助函数
# 权重初始化(卷积核初始化)
# tf.truncated_normal()不同于tf.random_normal(),返回的值中不会偏离均值两倍的标准差
# 参数shpae为一个列表对象,例如[5, 5, 1, 32]对应
# 5,5 表示卷积核的大小, 1代表通道channel,对彩色图片做卷积是3,单色灰度为1
# 最后一个数字32,卷积核的个数,(也就是卷基层提取的特征数量)
#   显式声明数据类型,切记
def weight_init(shape,name):
    weights = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights,name=name)

# 偏置的初始化
def biases_init(shape,name):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases,name=name)

# 随机选取mini_batch
# def get_random_batchdata(n_samples, batchsize):
#     start_index = np.random.randint(0, n_samples - batchsize)
#     return (start_index, start_index + batchsize)

# 全连接层权重初始化函数xavier
def xavier_init(layer1, layer2,name ,constant = 1):
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    return tf.Variable(tf.random_uniform((layer1, layer2), minval = Min, maxval = Max, dtype = tf.float32),name=name)

# 卷积
def conv2d(x, w,name):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME',name=name)
# 源码的位置在tensorflow/python/ops下nn_impl.py和nn_ops.py
# 这个函数接收两个参数,x 是图像的像素, w 是卷积核
# x 张量的维度[batch, height, width, channels]
# w 卷积核的维度[height, width, channels, channels_multiplier]
# tf.nn.conv2d()是一个二维卷积函数,
# stirdes 是卷积核移动的步长,4个1表示,在x张量维度的四个参数上移动步长
# padding 参数'SAME',表示对原始输入像素进行填充,卷积后映射的2D图像与原图大小相等
# 填充,是指在原图像素值矩阵周围填充0像素点
# 如果不进行填充,假设 原图为 32x32 的图像,卷积和大小为 5x5 ,卷积后映射图像大小 为 28x28


# 池化
def max_pool_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=name)


def max_pool_with_argmax(net, stride):
    '''
    重定义一个最大池化函数，返回最大池化结果以及每个最大值的位置(是个索引，形状和池化结果一致)

    args:
        net:输入数据 形状为[batch,in_height,in_width,in_channels]
        stride：步长，是一个int32类型，注意在最大池化操作中我们设置窗口大小和步长大小是一样的
    '''
    # 使用mask保存每个最大值的位置 这个函数只支持GPU操作
    _, mask = tf.nn.max_pool_with_argmax(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],
                                         padding='SAME')
    # 将反向传播的mask梯度计算停止
    mask = tf.stop_gradient(mask)
    # 计算最大池化操作
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    # 将池化结果和mask返回
    return net, mask


def un_max_pool(net,mask,stride=2):
    '''
    定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
    args:
        net:最大池化后的输出，形状为[batch, height, width, in_channels]
        mask：位置索引组数组，形状和net一样
        stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
    '''
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

def max_unpool_2x2(x):
    shape=x.get_shape().as_list()
    inference = tf.image.resize_nearest_neighbor(images=x,size=[shape[1]*2,shape[2]*2])
    return inference

#OctaveConv
def octaveconv(x,w,name="",alpha_in=0.25,alpha_out=0.25):

    shape_1=w.get_shape().as_list()
    print(shape_1)
    ch_in=int(w.shape[2])
    ch_out=int(w.shape[3])
    ch_in_4=int(int(x.shape[1])*0.25)
    ch_in_2=int(int(x.shape[1])*0.5)
    hf_ch_in=int(ch_in*(1-alpha_in))
    hf_ch_out=int(ch_out*(1-alpha_out))

    lf_ch_in=ch_in-hf_ch_in
    lf_ch_out=ch_out-hf_ch_out

    #高平输入，高平输出
    hf_data_in=x[:,:,:,0:hf_ch_in]
    hf_hf_out=w[:,:,0:hf_ch_in,0:hf_ch_out]
    hf_ch_out_1=tf.nn.conv2d(hf_data_in, hf_hf_out, strides=[1, 1, 1, 1], padding='SAME')


    #低平输入，低平输出
    lf_data_in = x[:, :, : , hf_ch_in:]
    lf_data_pool=tf.nn.max_pool(lf_data_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    lf_lf_out = w[:, :, hf_ch_in:, hf_ch_out:]
    lf_ch_out_1 = tf.nn.conv2d(lf_data_pool, lf_lf_out, strides=[1, 1, 1, 1], padding='SAME')

    #高平输入，低平输出
    # hf_data_in=x[:,:,:,0:hf_ch_in]
    hf_lf_data = tf.nn.max_pool(hf_data_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    hf_lf_out=w[:,:,0:hf_ch_in,hf_ch_out:]
    lf_ch_out_2=tf.nn.conv2d(hf_lf_data, hf_lf_out, strides=[1, 1, 1, 1], padding='SAME')


    #低平输入，高平输出
    # lf_data_in = x[:, :, : , hf_ch_in:]

    lf_hf_out = w[:, :, hf_ch_in:, 0:hf_ch_out]
    lf_ch_out_ = tf.nn.conv2d(lf_data_pool, lf_hf_out, strides=[1, 1, 1, 1], padding='SAME')
    hf_ch_out_2=max_unpool_2x2(lf_ch_out_)

    hf_out=hf_ch_out_1+hf_ch_out_2
    lf_out=lf_ch_out_1+lf_ch_out_2
    hf_out_final = tf.nn.max_pool(hf_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    out_data=tf.concat([hf_out_final,lf_out],axis=3)
    return out_data






# train_data_size=[1,1,1,1,1,1,1,1,1,1]
# test_data_size=[1,1,1,1,1,1,1,1,1,1]
# valid_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]

train_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
test_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
valid_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]

train_x,train_y=load_cifar_train(train_data_size,data_base_path="cifar10_class_data_wubiaoji")
test_x,test_y=load_cifar_test(test_data_size,data_base_path="cifar10_class_data_wubiaoji")
valid_x,valid_y=load_cifar_test(valid_data_size,data_base_path="cifar10_class_data_wubiaoji")
test_y_=np.array(test_y)
test_x_=np.array(test_x)
index_list = [i for i in range(train_y.shape[0])]
random.shuffle(index_list)
print(index_list)
# x 是手写图像的像素值,y 是图像对应的标签
x = tf.placeholder(tf.float32, [None, 32*32*3],name="x")
y = tf.placeholder(tf.float32, [None, 10],name="y")
# 把灰度图像一维向量,转换为28x28二维结构
x_image = tf.reshape(x, [-1, 32, 32, 3])
# -1表示任意数量的样本数,大小为28x28深度为一的张量
# 可以忽略(其实是用深度为28的,28x1的张量,来表示28x28深度为1的张量)



w_conv1 = weight_init([5, 5, 3, 16],name="w_conv1")        # 5x5,深度为1,16个
b_conv1 = biases_init([16],name="b_conv1")
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1,name="conv1") + b_conv1)    # 输出张量的尺寸:28x28x16
h_pool1 = max_pool_2x2(h_conv1,name="pool1")                                   # 池化后张量尺寸:14x14x16
# h_pool1 , 14x14的16个特征图


# w_conv2 = weight_init([5, 5, 16, 32],name="w_conv2")                             # 5x5,深度为16,32个
# b_conv2 = biases_init([32],name="b_conv2")
# h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2,name="conv2") + b_conv2)    # 输出张量的尺寸:14x14x32
# h_pool2 = max_pool_2x2(h_conv2,name="pool2")                                   # 池化后张量尺寸:7x7x32
# h_pool2 , 7x7的32个特征图

w_conv2 = weight_init([5, 5, 16, 32],name="w_conv1")        # 5x5,深度为1,16个
b_conv2 = biases_init([32],name="b_conv1")
h_conv2 = tf.nn.relu(octaveconv(h_pool1, w_conv2,name="conv1") + b_conv2)    # 输出张量的尺寸:28x28x16
# h_pool2 = max_pool_2x2(h_conv1,name="pool1")                                   # 池化后张量尺寸:14x14x16


w_conv3 = weight_init([5, 5, 32, 64],name="w_conv3")                             # 5x5,深度为16,32个
b_conv3 = biases_init([64],name="b_conv3")
h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3,name="conv3") + b_conv3)    # 输出张量的尺寸:14x14x32
h_pool3 = max_pool_2x2(h_conv3,name="pool3")

# h_pool2 , 7x7的32个特征图
w_conv4 = weight_init([5, 5,64, 128],name="w_conv4")                             # 5x5,深度为16,32个
b_conv4 = biases_init([128],name="b_conv4")
h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4,name="conv4") + b_conv4)    # 输出张量的尺寸:14x14x32
h_pool4 = max_pool_2x2(h_conv4,name="pool4")

# h_pool2是一个7x7x32的tensor,将其转换为一个一维的向量
h_fpool4 = tf.reshape(h_pool4, [-1, 2*2*128],name="h_fpool4")
# 全连接层,隐藏层节点为512个
# 权重初始化
w_fc1 = xavier_init(2*2*128, fch_nodes,name="w_fc1")
b_fc1 = biases_init([fch_nodes],name="b_fc1")
h_fc1 = tf.nn.relu(tf.matmul(h_fpool4, w_fc1) + b_fc1,name="fc1")

# 全连接隐藏层/输出层
# 为了防止网络出现过拟合的情况,对全连接隐藏层进行 Dropout(正则化)处理,在训练过程中随机的丢弃部分
# 节点的数据来防止过拟合.Dropout同把节点数据设置为0来丢弃一些特征值,仅在训练过程中,
# 预测的时候,仍使用全数据特征
# 传入丢弃节点数据的比例
#keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=drop_prob)

# 隐藏层与输出层权重初始化
w_fc2 = xavier_init(fch_nodes, 10,name="w_out")
b_fc2 = biases_init([10],name="b_out")

# 未激活的输出
y_ = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2,name="out")
# 激活后的输出
y_out = tf.nn.softmax(y_,name="out_final")

# #tensorboard xiangguan
# tf.summary.histogram("wd1", w_conv1)
# tf.summary.histogram("wd2", w_conv2)
# tf.summary.histogram("bd1", b_conv1)
# tf.summary.histogram("bd2", b_conv2)
# tf.summary.histogram("wfc1", w_fc1)
# tf.summary.histogram("bfc1", b_fc1)
# tf.summary.histogram("wout", w_fc2)
# tf.summary.histogram("bout", b_fc2)

# #tensorboard卷积核
# x_min=tf.reduce_min(w_conv1)
# x_max=tf.reduce_max(w_conv1)
# kernel_0_1=(w_conv1-x_min)/(x_max-x_min)
# kernel_transposed=tf.transpose(kernel_0_1,[3,0,1,2])
#
# x_min1=tf.reduce_min(w_conv2)
# x_max1=tf.reduce_max(w_conv2)
# kernel_0_11=(w_conv2-x_min1)/(x_max1-x_min1)
# kernel_transposed1=tf.transpose(kernel_0_11,[3,0,1,2])

# tf.summary.image("cov1/filters",kernel_transposed,max_outputs=5)
# tf.summary.image("cov2/filters",kernel_transposed1,max_outputs=5)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices = [1]),name="cross_entropy")

# tensorflow自带一个计算交叉熵的方法
# 输入没有进行非线性激活的输出值 和 对应真实标签
#cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))

# 优化器选择Adam(有多个选择)
optimizer = tf.train.AdamOptimizer(learning_rate,name="optimizer").minimize(cross_entropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate,name="optimizer").minimize(cross_entropy)
# 准确率
# 每个样本的预测结果是一个(1,10)的vector
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1),name="correct_prediction")
# tf.cast把bool值转换为浮点数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

# tf.summary.scalar("cost",cross_entropy)
# tf.summary.scalar("accuracy",accuracy)
#
# merged=tf.summary.merge_all()
# log_path="mnist_logs/logs7"

saver=tf.train.Saver(max_to_keep=1)

# 会话
with tf.Session() as sess:
    if os.path.exists('AE1/cifar_model/checkpoint'):
        model_file = tf.train.latest_checkpoint('AE1/cifar_model/')
        saver.restore(sess, model_file)
    else:
        # 全局变量进行初始化的Operation
        init = tf.global_variables_initializer()
        sess.run(init)
    # writer = tf.summary.FileWriter(log_path, sess.graph)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    print("network params is:",get_num_params())
    step=1
    Cost = []
    Accuracy = []
    for i in range(train_epochs):
        for j in range(batch_update):
            start,index_list,batch_x,batch_y = next_batch_(train_x,train_y, batch_size,start,index_list)
            _, cost, accu = sess.run([optimizer, cross_entropy,accuracy], feed_dict={x:batch_x, y:batch_y})
            # summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            # writer.add_summary(summary, step)
            step+=1
            Cost.append(cost)
            Accuracy.append(accu)
        if i % display_step ==0:
            #经过该循环可以得出信息，两次同样的计算中存在误差，而此误差是来自机器计算中产生的，无法避免，属于副点数运算误差
            # for j in range(2):
            cost, accu = sess.run([cross_entropy,accuracy], feed_dict={x:valid_x, y:valid_y})
            print ('Epoch : %d ,  Cost : %.7f'%(i+1, cost))
            print('Epoch : %d ,  accuracy : %.7f' % (i + 1, accu))
            saver.save(sess,"AE1/cifar_model/model.ckpt")
    print('training finished')
    _, cost, accu = sess.run([optimizer, cross_entropy, accuracy], feed_dict={x: test_x, y: test_y})
    print("final cost:",cost)
    print("final accu:",accu)

