import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import random
import os
import time

from layers import *
from model import *

img_height = 256
img_width = 256
img_layer = 3

batch_size = 1
pool_size = 50
max_images = 300

to_restore = False
save_training_images = False
to_train = False
to_test = True
out_path = "./output"
check_dir = "./output/checkpoints/"


class CycleGAN():

    def input_setup(self):
        filename_A = tf.train.match_filenames_once("./Japanese_after_rotate/*.jpg")
        self.queue_length_A = tf.size(filename_A)
        filename_B = tf.train.match_filenames_once("./smokey_after_rotate/*.jpg")
        self.queue_length_B = tf.size(filename_B)

        filename_A_queue = tf.train.string_input_producer(filename_A)
        filename_B_queue = tf.train.string_input_producer(filename_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_A_queue)
        _, image_file_B = image_reader.read(filename_B_queue)
        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)


    def input_read(self,sess):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_file_A = sess.run(self.queue_length_A)
        num_file_B = sess.run(self.queue_length_B)

        self.fake_images_A = np.zeros((pool_size,1,img_height,img_width,img_layer))
        self.fake_images_B = np.zeros((pool_size,1,img_height,img_width,img_layer))

        self.A_input = np.zeros((max_images,batch_size,img_height,img_width,img_layer))
        self.B_input = np.zeros((max_images,batch_size,img_height,img_width,img_layer))

        for i in range(max_images):
            image_tensor = sess.run(self.image_A)
            if image_tensor.size==img_width*img_height*img_layer:
                self.A_input[i] = image_tensor.reshape((batch_size,img_height,img_width,img_layer))

        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            if image_tensor.size==img_width*img_height*img_layer:
                self.B_input[i] = image_tensor.reshape((batch_size,img_height,img_width,img_layer))

        coord.request_stop()
        coord.join(threads)


    def model_setup(self):
        self.input_A = tf.placeholder(tf.float32,[batch_size,img_height,img_width,img_layer],name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer], name="input_B")

        self.fake_pool_A = tf.placeholder(tf.float32,[None,img_height,img_width,img_layer],name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0,trainable=False,name="global_step")
        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32,shape=[],name="lr")

        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_9blocks(self.input_A,name="g_A")
            self.fake_A = build_generator_resnet_9blocks(self.input_B,name="g_B")

            self.rec_A = build_gen_discriminator(self.input_A,"d_A")
            self.rec_B = build_gen_discriminator(self.input_B,"d_B")

            scope.reuse_variables()

            self.fake_rec_A = build_gen_discriminator(self.fake_A,"d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B,"d_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B,"g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A,"g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A,"d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B,"d_B")


    def loss_cals(self):
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A))+tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        g_loss_A = cyc_loss*10 +disc_loss_B
        g_loss_B = cyc_loss*10+disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A))+tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(
            tf.squared_difference(self.rec_B, 1))) / 2

        optimizer = tf.train.AdamOptimizer(self.lr,beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if "d_A" in var.name]
        d_B_vars = [var for var in self.model_vars if "d_B" in var.name]
        g_A_vars = [var for var in self.model_vars if "g_A" in var.name]
        g_B_vars = [var for var in self.model_vars if "g_B" in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A,var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B,var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A,var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B,var_list=g_B_vars)

        for var in self.model_vars:
            print(var.name)

        self.g_A_loss = tf.summary.scalar("g_A_loss",g_loss_A)
        self.g_B_loss = tf.summary.scalar("g_B_loss",g_loss_B)
        self.d_A_loss = tf.summary.scalar("d_A_loss",d_loss_A)
        self.d_B_loss = tf.summary.scalar("d_B_loss",d_loss_B)


    def save_training_images(self,sess,epoch):
        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0,10):
            fake_A_temp,fake_B_temp,cyc_A_temp,cyc_B_temp = sess.run([self.fake_A,self.fake_B,self.cyc_A,self.cyc_B],feed_dict={
                self.input_A:self.A_input[i],
                self.input_B:self.B_input[i]
            })
            imsave("./output/imgs/fakeA_"+str(epoch)+"_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/fakeB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/cycA_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((cyc_A_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/cycB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((cyc_B_temp[0] + 1) * 127.5).astype(np.uint8))


    def fake_image_pool(self,num_fake,fake,fake_pool):
        if num_fake<pool_size:
            fake_pool[num_fake] = fake
            return fake
        else:
            p = random.random()
            if p>0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake


    def train(self):
        self.input_setup()
        self.model_setup()
        self.loss_cals()

        init = [tf.local_variables_initializer(),tf.global_variables_initializer()]
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            self.input_read(sess)

            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess,chkpt_fname)

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            for epoch in range(sess.run(self.global_step),500):
                print("In the epoch",epoch)
                saver.save(sess,os.path.join(check_dir,"cycleGAN"),global_step=epoch)

                if epoch<100:
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002-0.0002*(epoch-100)/400

                if save_training_images:
                    self.save_training_images(sess,epoch)

                for ptr in range(max_images):
                    print("In the iteration",ptr)
                    print("Starting",time.ctime())

                    # optimize G_A
                    _, summary_str, fake_B_temp = sess.run(
                        [self.g_A_trainer, self.g_A_loss, self.fake_B],
                        feed_dict={self.input_A:self.A_input[ptr],self.input_B:self.B_input[ptr],self.lr:curr_lr}
                    )
                    writer.add_summary(summary_str,epoch*max_images+ptr)
                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs,fake_B_temp,self.fake_images_B)

                    # optimize D_B
                    _, summary_str = sess.run([self.d_B_trainer,self.d_B_loss],feed_dict={
                        self.input_A:self.A_input[ptr],
                        self.input_B:self.B_input[ptr],
                        self.lr:curr_lr,
                        self.fake_pool_B:fake_B_temp1
                    })
                    writer.add_summary(summary_str,epoch*max_images+ptr)

                    # optimize G_B
                    _,summary_str,fake_A_temp = sess.run([self.g_B_trainer,self.g_B_loss,self.fake_A],feed_dict={
                        self.input_A: self.A_input[ptr],
                        self.input_B: self.B_input[ptr],
                        self.lr: curr_lr
                    })
                    writer.add_summary(summary_str,epoch*max_images+ptr)
                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs,fake_A_temp,self.fake_images_A)
                    # optimize D_A
                    _,summary_str = sess.run([self.d_A_trainer,self.d_A_loss],feed_dict={
                        self.input_A: self.A_input[ptr],
                        self.input_B: self.B_input[ptr],
                        self.lr: curr_lr,
                        self.fake_pool_A: fake_A_temp1
                    })
                    writer.add_summary(summary_str,epoch*max_images+ptr)

                    self.num_fake_inputs+=1

                sess.run(tf.assign(self.global_step,epoch+1))

            writer.add_graph(sess.graph)


    def test(self):
        print("Testing the results")

        self.input_setup()
        self.model_setup()
        saver = tf.train.Saver()
        init = [tf.local_variables_initializer(),tf.global_variables_initializer()]

        with tf.Session() as sess:
            sess.run(init)
            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess,chkpt_fname)

            if not os.path.exists("./output/imgs/test"):
                os.makedirs("./output/imgs/test")

            for i in range(max_images):
                fake_A_temp,fake_B_temp = sess.run([self.fake_A,self.fake_B],feed_dict={
                    self.input_A:self.A_input[i],
                    self.input_B:self.B_input[i]
                })
                imsave("./output/imgs/fakeA_" + str(i) + ".jpg",
                       ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/imgs/fakeB_" + str(i) + ".jpg",
                       ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))


def main():
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()


if __name__ == '__main__':
    main()
