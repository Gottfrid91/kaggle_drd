import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.transform import resize
import cv2
import itertools

def plot_confusion_matrix(cls_pred, cls_true, num_classes):
    '''
    :param cls_pred: numpy array with class prediction
    :param cls_true: numpy array with class labels
    :param num_classes: constant depicting number of classes
    :return: this function saves a plot of the confusion matrix in the working dir
    '''

    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.savefig('./tmp/graphs/' + 'confusion_matrix.jpeg')

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def plot_confusion_matrix_2(cls_pred, cls_true, class_names= range(0,4),
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        norm = "normalized"
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        plt.savefig('./tmp/graphs/' + 'confusion_matrix_' + 'norm' + '.jpeg')
    else:
        plt.savefig('./tmp/graphs/' + 'confusion_matrix_.jpeg')

def quadratic_kappa(y, t, eps=1e-15):
    '''

    :param y: labels
    :param t: predictions
    :param eps: default_values
    :return: The weighted kappa metric
    '''
    n_y_values = np.max(y) + 1
    n_t_values = np.max(t) + 1
    n = max(n_y_values, n_t_values)
    y_onehot = np.eye(n)[y]
    t_onehot = np.eye(n)[t]

    # Assuming y and t are one-hot encoded!
    num_scored_items = y_onehot.shape[0]

    num_ratings = y_onehot.shape[1]
    ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
                          reps=(1, num_ratings))
    ratings_squared = (ratings_mat - ratings_mat.T) ** 2
    weights = ratings_squared / (float(num_ratings) - 1) ** 2

    # We norm for consistency with other variations.
    y_norm = y_onehot / (eps + y_onehot.sum(axis=1)[:, None])

    # The histograms of the raters.
    hist_rater_a = y_norm.sum(axis=0)
    hist_rater_b = t_onehot.sum(axis=0)

    # The confusion matrix.
    conf_mat = np.dot(y_norm.T, t_onehot)

    # The nominator.
    nom = np.sum(weights * conf_mat)
    expected_probs = np.dot(hist_rater_a[:, None],
                            hist_rater_b[None, :])
    # The denominator.
    denom = np.sum(weights * expected_probs / num_scored_items)

    return 1 - nom / denom

def visualize_gradients(image, conv_output, conv_grad, gb_viz):
    '''

    :param image: numpy image of shape NxHxWxC
    :param conv_output: target_conv_layer, numpy
    :param conv_grad:  target_conv_layer_grad, numpy
    :param gb_viz: gb_grad, numpy
    :return: return displays of three images, CAM
    '''
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)

    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (448, 448), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(131)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')

    gb_viz = np.dstack((
        gb_viz[:, :, 0],
        gb_viz[:, :, 1],
        gb_viz[:, :, 2],
    ))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = fig.add_subplot(132)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')

    gd_gb = np.dstack((
        gb_viz[:, :, 0] * cam,
        gb_viz[:, :, 1] * cam,
        gb_viz[:, :, 2] * cam,
    ))
    ax = fig.add_subplot(133)
    imgplot = plt.imshow(gd_gb)
    ax.set_title('guided Grad-CAM')

    plt.show()