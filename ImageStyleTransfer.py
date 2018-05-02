
# coding: utf-8

# In[2]:


from keras.preprocessing.image import load_img, img_to_array
target_image_path = 'train/NewYork.jpg'
style_reference_image_path = 'style/Van_Gogh.jpg'


# In[3]:


width, height = load_img(target_image_path).size
img_height = 800
img_width = int(width * img_height / height)


# In[5]:


import numpy as np
from keras.applications import vgg19
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# In[6]:


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[7]:
import os
import tensorflow as tf
def get_session(gpu_fraction=0.3):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
import keras.backend.tensorflow_backend as KTF
KTF.set_session(get_session())
target_image = KTF.constant(preprocess_image(target_image_path))
style_reference_image = KTF.constant(preprocess_image(style_reference_image_path))
combination_image = KTF.placeholder((1, img_height, img_width, 3))


# In[9]:


input_tensor = KTF.concatenate([target_image, style_reference_image,
                              combination_image], axis=0)


# In[10]:


model = vgg19.VGG19(input_tensor=input_tensor, 
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')


# In[11]:


def content_loss(base, combination):
    return KTF.sum(KTF.square(combination - base))


# In[12]:


def gram_matrix(x):
    features = KTF.batch_flatten(KTF.permute_dimensions(x, (2, 0, 1)))
    gram = KTF.dot(features, KTF.transpose(features))
    return gram


# In[13]:


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return KTF.sum(KTF.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# In[14]:


def total_variation_loss(x):
    a = KTF.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = KTF.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return KTF.sum(KTF.pow(a + b, 1.25))


# In[15]:


outputs_dict = dict([(layer.name, layer.output) for layer in model.layers]) 
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025


# In[16]:


loss = KTF.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)


# In[17]:



for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl


# In[18]:


loss += total_variation_weight * total_variation_loss(combination_image)


# In[19]:


grads = KTF.gradients(loss, combination_image)[0]
fetch_loss_and_grads = KTF.function([combination_image], [loss, grads])


# In[20]:


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
evaluator = Evaluator()


# In[21]:


from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time


# In[22]:


result_prefix = style_reference_image_path.replace('.jpg','')
iterations = 50


# In[ ]:


x = preprocess_image(target_image_path) 
x= x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_iteration%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

