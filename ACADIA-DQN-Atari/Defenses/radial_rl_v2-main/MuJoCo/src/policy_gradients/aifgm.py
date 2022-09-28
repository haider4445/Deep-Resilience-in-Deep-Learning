



      
beta1 = 0.99
beta2 = 0.999
num_iter1 = FLAGS.num_iter
weight=0
t = np.arange(1,num_iter1+0.1,1)
y1 = np.sqrt(1 - beta2**t) / (1 - beta1**t)

for x1 in y1:
    weight+=x1
      


def graph(x, y, i, x_max, x_min, grad, grad2):
  eps = 2.0 * 16 / 255.0
  alpha_norm2 = eps * np.sqrt(299 * 299 * 3)


  delta = 1e-08
  num_classes = 1001

      
  pred = tf.argmax(end_points_v3['Predictions'], 1)

  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)  

  logits = logits_v3
  cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(one_hot,logits,label_smoothing=0.0,weights=1.0)

      
  auxlogits = end_points_v3['AuxLogits']
      
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)
  

  noise = tf.gradients(cross_entropy, x)[0]


  

  noise = noise / tf.reduce_sum(tf.abs(noise), [1,2,3], keep_dims=True)
  noise2 = beta2 * grad2 + (1-beta2) * tf.square(noise)
  
  noise = beta1 * grad + (1-beta1) * noise
  alpha_t = alpha_norm2 * num_iter * (tf.sqrt(1-beta2**(i+1)) / (1-beta1**(i+1))) / weight
  
  normalized_grad = noise/(tf.sqrt(noise2)+delta)
#  normalized_grad = noise
  square = tf.reduce_sum(tf.square(normalized_grad),reduction_indices=[1,2,3],keep_dims=True)
  normalized_grad = normalized_grad / tf.sqrt(square)
  x = x + alpha_t * normalized_grad
  
  
  
  
  
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise, noise2

        