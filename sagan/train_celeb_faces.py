import tensorflow as tf
tf.enable_eager_execution()
from utils import tf_record_parser, normalizer
from utils import save_model

print(tf.__version__)

tfe = tf.contrib.eager

import os
from celeb.model import Generator, Discriminator
from libs.loss import discriminator_loss, generator_loss

z_size = 128
learning_rate_generator = 0.0001
learning_rate_discriminator = 0.0004
batch_size = 64
alpha = 0.1
beta1 = 0.0
beta2 = 0.9
total_train_steps = 400000
dtype = tf.float64

train_dataset = tf.data.TFRecordDataset(["/home/thalles/Documents/datasets/celeb_faces/tfrecords/train.tfrecords"])
train_dataset = train_dataset.map(tf_record_parser, num_parallel_calls=4)
train_dataset = train_dataset.map(normalizer, num_parallel_calls=4)
train_dataset = train_dataset.shuffle(1000)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(batch_size)

generator_net = Generator(dtype=dtype)
discriminator_net = Discriminator(alpha=alpha, dtype=dtype)

generator_net = Generator()
discriminator_net = Discriminator(alpha=alpha)

basepath = "./celeb/checkpoints"
logdir = os.path.join(basepath, "logs")
tf_board_writer = tf.contrib.summary.create_file_writer(logdir)
tf_board_writer.set_as_default()

generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_generator, beta1=beta1, beta2=beta2)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_discriminator, beta1=beta1, beta2=beta2)

global_step = tf.train.get_or_create_global_step()

gen_checkpoint_dir = os.path.join(basepath, "generator")
gen_checkpoint_prefix = os.path.join(gen_checkpoint_dir, "model.ckpt")
gen_root = tfe.Checkpoint(optimizer=generator_optimizer,
                          model=generator_net,
                          optimizer_step=global_step)

disc_checkpoint_dir = os.path.join(basepath, "discriminator")
disc_checkpoint_prefix = os.path.join(disc_checkpoint_dir, "model.ckpt")
disc_root = tfe.Checkpoint(optimizer=discriminator_optimizer,
                           model=discriminator_net,
                           optimizer_step=global_step)

if os.path.exists(basepath):
    try:
        gen_root.restore(tf.train.latest_checkpoint(gen_checkpoint_dir))
        print("Generator model restored")
    except Exception as ex:
        print("Error loading the Generator model:", ex)

    try:
        disc_root.restore(tf.train.latest_checkpoint(disc_checkpoint_dir))
        print("Discriminator model restored")
    except Exception as ex:
        print("Error loading the Disciminator model:", ex)
    print("Current global step:", tf.train.get_or_create_global_step().numpy())
else:
    print("Model folder not found.")

# generate sample noise for evaluation
fake_input_test = tf.random_normal(shape=(12, z_size), dtype=dtype)

for _, (batch_real_images) in enumerate(train_dataset):

    fake_input = tf.random_normal(shape=(batch_size, z_size), dtype=dtype)

    with tf.contrib.summary.record_summaries_every_n_global_steps(500):

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # run the generator with the random noise batch
            g_model = generator_net(fake_input, is_training=True)

            # run the discriminator with real input images
            d_logits_real = discriminator_net(batch_real_images, is_training=True)

            # run the discriminator with fake input images (images from the generator)
            d_logits_fake = discriminator_net(g_model, is_training=True)

            # compute the generator loss
            gen_loss = generator_loss(d_logits_fake)

            # compute the discriminator loss
            dis_loss = discriminator_loss(d_logits_real, d_logits_fake)

            # tf.contrib.summary.scalar('generator_loss', gen_loss)
            # tf.contrib.summary.scalar('discriminator_loss', dis_loss)
            tf.contrib.summary.image('generator_image', tf.to_float(g_model), max_images=5)

        # get all the discriminator variables, including the tfe variables
        discriminator_variables = discriminator_net.variables
        discriminator_variables.append(discriminator_net.attention.gamma)

        discriminator_grads = d_tape.gradient(dis_loss, discriminator_variables)

        # get all the discriminator variables, including the tfe variables
        generator_variables = generator_net.variables
        generator_variables.append(generator_net.attention.gamma)

        generator_grads = g_tape.gradient(gen_loss, generator_variables)

        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator_variables),
                                                global_step=global_step)

        generator_optimizer.apply_gradients(zip(generator_grads, generator_variables),
                                            global_step=global_step)

        # register the Generator and Discriminator's gradients to tensorboard
        for g, var in zip(discriminator_grads, discriminator_variables):
            tf.contrib.summary.histogram(var.name, g)

        for g, var in zip(generator_grads, generator_variables):
            if g is None:
                continue
            tf.contrib.summary.histogram(var.name, g)

    counter = global_step.numpy()

    if counter % 2000 == 0:
        print("Current step:", counter)
        with tf.contrib.summary.always_record_summaries():
            generated_samples = generator_net(fake_input_test, is_training=False)
            tf.contrib.summary.image('test_generator_image', tf.to_float(generated_samples), max_images=12)

    if counter % 20000 == 0:
        # save and download the mode
        save_model(gen_root, gen_checkpoint_prefix)
        save_model(disc_root, disc_checkpoint_prefix)

    if counter >= total_train_steps:
        save_model(gen_root, gen_checkpoint_prefix)
        save_model(disc_root, disc_checkpoint_prefix)
        break
