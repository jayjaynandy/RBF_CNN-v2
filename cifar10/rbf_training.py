import tensorflow as tf
import numpy as np
import setup_data
import matplotlib.pyplot as plt
sess = tf.Session()
import os

#%% parameters

w,h,c,nb_filter = 3,3,3,64    ## RBF filter configurations. choose desired number of files.
my_dir = './layer/lp_'+str(nb_filter)+'/'

if not os.path.exists(my_dir):
  os.makedirs(my_dir)

batch_size = 250
_s = 32
ch = 3
data = setup_data.Cifar().train_data
epochs = 50
total_batch = int(data.shape[0]/ batch_size)
LR = 0.01


def save_images(name, _imgs, r, c):
    _imgs = np.clip(_imgs, 0, 1)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(_imgs[cnt, :,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(my_dir+"images_" + str(name) +".png")
    plt.close()

#%%
tf_x = tf.placeholder(tf.float32, [batch_size, _s, _s, ch])
tf_lr = tf.placeholder(tf.float32, [])
tf_mean = tf.Variable(tf.random_normal([w,h,c,nb_filter], mean = 0.5, stddev=0.2))
tf_covs = tf.Variable(tf.random_normal([nb_filter], mean = 0.5, stddev = 0.01))

tf_ones = tf.ones((3,3,ch,1))    
tensor1 = tf.nn.conv2d(tf_x, tf_mean, strides=(1,1,1,1), padding="VALID")
tensor2 = tf.multiply(tf_x, tf_x, name=None)
tensor2 = tf.nn.conv2d(tensor2, tf_ones, strides=(1,1,1,1), padding="VALID")    
tensor3 = tf.multiply(tf_mean, tf_mean, name=None)
tensor3 = tf.reduce_sum(tensor3, [0,1,2]) 
out = 2*tensor1 - tensor2 - tensor3
out = out/(2*tf.exp(tf_covs)+ 1e-25) - tf_covs*0.5
#out = tf.nn.sigmoid(out)

loss = -tf.reduce_mean(tf.reduce_max(out, axis=[3]))
####################### reconstruction #######################

optim = tf.train.AdamOptimizer(tf_lr)
train = optim.minimize(loss)

init_op = tf.initialize_all_variables()
sess.run(init_op)

for e in range(epochs):
    print('Printing means: epoch ... ' + str(e) )
    np_means = sess.run(tf_mean)
    np_covs = sess.run(tf_covs)
    
    print(sess.run(tf_covs))
    means_viz = np.transpose(np_means, axes= (3,0,1,2))
    save_images(e, means_viz, 5, 10)
    np.save(my_dir+'mean3x3', np_means)
    np.save(my_dir+'covs3x3', np_covs)    
    LR *= 0.75
    
    for d in range(total_batch-1):
        x_batch = np.copy(data[d*batch_size:(d+1)*batch_size])
        _, np_loss, np_mean = sess.run([train, loss, tf_mean], {tf_x: x_batch, tf_lr: LR})
        
        if d % 50 ==0:
            print(str(d)+ '  : ' + str(np_loss))
            means_viz = np.transpose(np_mean, axes= (3,0,1,2))
            save_images(e, means_viz, 5, 10)
