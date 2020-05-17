# %%
from IPython import get_ipython
from glob import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import pandas as pd

# %% [markdown]
# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.
# %% [markdown]
# ---
# ## Step 0: Load The Data

# %%
# Load pickled data

# TODO: Fill this in based on where you saved the training and testing data

training_file = './data/train.p'
validation_file='./data/valid.p'
testing_file = './data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# %% [markdown]
# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 
# %% [markdown]
# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# %%

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid)

# Number of testing examples
n_test = len(X_test)

# Shape of a traffic sign image
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# %% [markdown]
# ### Include an exploratory visualization of the dataset
# %% [markdown]
# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# %% 
sign_names = pd.read_csv('signnames.csv', delimiter=',')
sign_names_dict = sign_names.to_dict()['SignName']
def signName(label):
    return '{}-{}'.format(label, sign_names_dict[label])

def signNames(labels):
    return list(map(lambda l: signName(l), labels))
# %%

get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(X_train[0])
print(signName(y_train[0]))

# %%
# Show random image from a train set
index = np.random.randint(0, len(X_train))
plt.imshow(X_train[index])
print(signName(y_train[index]))

# %% [markdown]
# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# %% [markdown]
# ### Pre-process the Data Set (normalization, grayscale, etc.)
# %% [markdown]
# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 

# %%
### Preprocessing data
def preprocess(x):
    return (x.astype(np.float32) - 128) / 128
X_train_processed = preprocess(X_train)
X_valid_processed = preprocess(X_valid)
X_test_processsed = preprocess(X_test)


# %%
# Sanity check for preprocessing step
print('Example red channel pixel value from first image {}'.format(X_train_processed[0][0][0][0]))
minV = print('Max value in preprocessed data {}'.format(np.max(X_train_processed)))
maxV = print('Min value in preprocessed data {}'.format(np.min(X_train_processed)))

# %% [markdown]
# ### Model Architecture

# %%

def conv2d(x, W, b, strides=1):
    conv = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='VALID')
    conv = tf.nn.bias_add(conv, b)
    return tf.nn.relu(conv)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

EPOCHS = 10
BATCH_SIZE = 128

dropout_keep_prob = tf.placeholder(tf.float32)
conv_dropout_keep_prob = tf.placeholder(tf.float32)

conv1 = None
conv2 = None
fc1 = None
fc2 = None
def ConvNet(x):
    global conv1
    global conv2
    global fc1
    global fc2

    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    wc1 = tf.Variable(tf.truncated_normal([5, 5, 3, 6], mu, sigma))
    bc1 = tf.Variable(tf.truncated_normal([6], mu, sigma))
    conv1 = conv2d(x, wc1, bc1, strides=1)

    # Pooling. Input = 28x28x6. Output = 14x14x6
    conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, keep_prob=conv_dropout_keep_prob)
    
    # Layer 2: Convolutional. Output = 10x10x16.
    wc2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma))
    bc2 = tf.Variable(tf.truncated_normal([16], mu, sigma))
    conv2 = conv2d(conv1, wc2, bc2, strides=1)

    # Pooling. Input = 10x10x16. Output = 5x5x16
    conv2 = maxpool2d(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, keep_prob=conv_dropout_keep_prob)

    # Flatten. Input = 5x5x16. Output = 400.
    conv2Flat = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    wd1 = tf.Variable(tf.truncated_normal([400, 120], mu, sigma))
    bd1 = tf.Variable(tf.truncated_normal([120], mu, sigma))
    fc1 = tf.add(tf.matmul(conv2Flat, wd1), bd1)

    # ReLU Activation
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout_keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    wd2 = tf.Variable(tf.truncated_normal([120, 84], mu, sigma))
    bd2 = tf.Variable(tf.truncated_normal([84], mu, sigma))
    fc2 = tf.add(tf.matmul(fc1, wd2), bd2)
    
    # ReLU Activation
    fc2 = tf.nn.relu(fc2)
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob=dropout_keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes
    wOut = tf.Variable(tf.truncated_normal([84, n_classes], mu, sigma))
    bOut = tf.Variable(tf.truncated_normal([n_classes], mu, sigma))
    logits = tf.add(tf.matmul(fc2, wOut), bOut)
    
    return logits


# %%
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = ConvNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss)

### Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    sess = tf.get_default_session()
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        maxI = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:maxI], y_data[offset:maxI]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, dropout_keep_prob: 1, conv_dropout_keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

saver = tf.train.Saver()

# %% [markdown]
# ### Train, Validate and Test the Model
# %% [markdown]
# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# %%

### Model Training
tf.set_random_seed(123456)
model_save_file = './model/model.ckpt'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_examples = len(X_train_processed)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_shuffled, y_shuffled = shuffle(X_train_processed, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_shuffled[offset:offset+BATCH_SIZE], y_shuffled[offset:offset+BATCH_SIZE]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, dropout_keep_prob: 0.5, conv_dropout_keep_prob: 0.9})
        
        validation_accuracy = evaluate(X_valid_processed, y_valid)
        print("EPOCH {}: Validation accuracy {:.3f}".format(i+1, validation_accuracy))

    saver.save(sess, model_save_file)
    print("Model saved")

# %% [markdown]
# ## Validation Accuracy
# %%
with tf.Session() as sess:
    saver.restore(sess, model_save_file)

    test_accuracy = evaluate(X_valid_processed, y_valid)
    print("Validation Accuracy = {:.3f}".format(test_accuracy))

# %% [markdown]
# ## Test the Model on Test Set
# %%
with tf.Session() as sess:
    saver.restore(sess, model_save_file)

    test_accuracy = evaluate(X_test_processsed, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

# %% [markdown]
# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.
# %% [markdown]
# ### Load and Output the Images

#%%

filenames = glob('./images/*')
test_imgs = []
test_labels = []
plt.figure(1, figsize=(15,15))
for i in range(len(filenames)):
    img = mpimg.imread(filenames[i])
    img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    test_imgs.append(img)
    label = int(filenames[i].split('/')[-1].split('_')[0])
    test_labels.append(label)
    # plot
    plt.subplot(2,3,i+1)
    plt.title(signName(label))
    plt.imshow(img)

# %% [markdown]
# ### Predict the Sign Type for Each Image

# %% 

test_imgs_processed = preprocess(np.array(test_imgs))
with tf.Session() as sess:
    saver.restore(sess, model_save_file)

    accuracy_test_imgs = evaluate(test_imgs_processed, test_labels)
    print("Accuracy = {:.2f}%".format(accuracy_test_imgs*100))
    print("======================")

    softmax = tf.nn.softmax(logits)
    top5_softmax_eval = sess.run(tf.nn.top_k(softmax, k=5), feed_dict={x: test_imgs_processed, y: test_labels, dropout_keep_prob: 1., conv_dropout_keep_prob: 1.0})
    print(top5_softmax_eval)

print("======================")
predictions = top5_softmax_eval.indices[:,0]
print("LABELS")
print(signNames(test_labels))
print("======================")
print("PREDICTIONS")
print(signNames(predictions))

# %% 

softmax_labels_named = list(map(lambda x: signNames(x), top5_softmax_eval.indices))
plt.figure(figsize=(10,20))
for i in range(len(softmax_labels_named)):
    plt.subplot(5,2,i*2+1)
    y_pos = np.arange(len(softmax_labels_named[i]))
    plt.barh(y_pos, top5_softmax_eval.values[i], align='center', alpha=0.5)
    plt.yticks(y_pos, softmax_labels_named[i])
    plt.subplot(5,2,i*2+2)
    plt.imshow(test_imgs[i])
plt.show()
# %% [markdown]
# ### Analyze Performance

# %% [markdown]
# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web
# %% [markdown]
# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


# %% [markdown]
# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
# %% [markdown]
# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# %% [markdown]
# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# %%
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input, dropout_keep_prob: 1., conv_dropout_keep_prob: 1.})
    print(activation.shape)
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

with tf.Session() as sess:
    saver.restore(sess, model_save_file)
    outputFeatureMap([test_imgs_processed[-1]], conv1)

# %%
