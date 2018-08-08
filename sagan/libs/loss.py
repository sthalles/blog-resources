import tensorflow as tf

def discriminator_loss(d_logits_real, d_logits_fake):
    # Hinge loss
    real_loss = tf.reduce_mean(tf.nn.relu(1. - d_logits_real))
    fake_loss = tf.reduce_mean(tf.nn.relu(1. + d_logits_fake))
    return real_loss + fake_loss

def generator_loss(d_logits_fake):
  return - tf.reduce_mean(d_logits_fake)
