#coding=utf-8
"""
PGCNet batch data generator
two different type input :point cloud and multi-view image
__author__ = Cush shen
"""

import numpy as np
from tqdm import tqdm
import h5py
import time
import tensorflow as tf


image_color_gray = 158
image_color_white = 255


def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def loadDataFile(filename):
    return load_h5(filename)


def get_model_learning_rate(
        learning_policy, base_learning_rate, learning_rate_decay_step,
        learning_rate_decay_factor, training_number_of_steps, learning_power,
        slow_start_step, slow_start_learning_rate):
    """Gets model's learning rate.

    Computes the model's learning rate for different learning policy.
    Right now, only "step" and "poly" are supported.
    (1) The learning policy for "step" is computed as follows:
      current_learning_rate = base_learning_rate *
        learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
    See tf.train.exponential_decay for details.
    (2) The learning policy for "poly" is computed as follows:
      current_learning_rate = base_learning_rate *
        (1 - global_step / training_number_of_steps) ^ learning_power

    Args:
      learning_policy: Learning rate policy for training.
      base_learning_rate: The base learning rate for model training.
      learning_rate_decay_step: Decay the base learning rate at a fixed step.
      learning_rate_decay_factor: The rate to decay the base learning rate.
      training_number_of_steps: Number of steps for training.
      learning_power: Power used for 'poly' learning policy.
      slow_start_step: Training model with small learning rate for the first
        few steps.
      slow_start_learning_rate: The learning rate employed during slow start.

    Returns:
      Learning rate for the specified learning policy.

    Raises:
      ValueError: If learning policy is not recognized.
    """
    global_step = tf.train.get_or_create_global_step()
    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(
            base_learning_rate,
            global_step,
            training_number_of_steps,
            end_learning_rate=0,
            power=learning_power)
    else:
        raise ValueError('Unknown learning policy.')

    return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                    learning_rate)


def _gather_loss(regularization_losses, scope):
    """
    Gather the loss.
    Args:
      regularization_losses: Possibly empty list of regularization_losses
        to add to the losses.
    Returns:
      A tensor for the total loss.  Can be None.
    """

    sum_loss = None
    # Individual components of the loss that will need summaries.
    loss = None
    regularization_loss = None

    # Compute and aggregate losses on the clone device.
    all_losses = []
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    if losses:
        loss = tf.add_n(losses, name='losses')
        all_losses.append(loss)
    if regularization_losses:
        regularization_loss = tf.add_n(regularization_losses,
                                       name='regularization_loss')
        all_losses.append(regularization_loss)
    if all_losses:
        sum_loss = tf.add_n(all_losses)

    # Add the summaries out of the clone device block.
    if loss is not None:
        tf.summary.scalar('/'.join(filter(None, ['Losses', 'loss'])), loss)
    if regularization_loss is not None:
        tf.summary.scalar('Losses/regularization_loss', regularization_loss)
    return sum_loss


def _optimize(optimizer, regularization_losses, scope, **kwargs):
    """
    Compute losses and gradients.
    Args:
      optimizer: A tf.Optimizer  object.
      regularization_losses: Possibly empty list of regularization_losses
        to add to the losses.
      **kwargs: Dict of kwarg to pass to compute_gradients().

    Returns:
      A tuple (loss, grads_and_vars).
        - loss: A tensor for the total loss.  Can be None.
        - grads_and_vars: List of (gradient, variable). Can be empty.
    """
    sum_loss = _gather_loss(regularization_losses, scope)
    grad = None
    if sum_loss is not None:
        grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return sum_loss, grad


def _gradients(grad):
    """
    Calculate the sum gradient for each shared variable across all clones.
    This function assumes that the grad has been scaled appropriately by
    1 / num_clones.

    Args:
      grad: A List of List of tuples (gradient, variable)

    Returns:
       tuples of (gradient, variable)
    """
    sum_grads = []
    for grad_and_vars in zip(*grad):
        # Note that each grad_and_vars looks like the following:
        #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(g)
        if grads:
            if len(grads) > 1:
                sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))

    return sum_grads


def optimize(optimizer, scope=None, regularization_losses=None, **kwargs):
    """
    Compute losses and gradients
    # Note: The regularization_losses are added to losses.

    Args:
     optimizer: An `Optimizer` object.
     regularization_losses: Optional list of regularization losses. If None it
       will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to
       exclude them.
     **kwargs: Optional list of keyword arguments to pass to `compute_gradients`.

    Returns:
     A tuple (total_loss, grads_and_vars).
       - total_loss: A Tensor containing the average of the losses including
         the regularization loss.
       - grads_and_vars: A List of tuples (gradient, variable) containing the sum
         of the gradients for each variable.
    """
    grads_and_vars = []
    losses = []
    if regularization_losses is None:
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope)
    # with tf.name_scope(scope):
    loss, grad = _optimize(optimizer,
                           regularization_losses,
                           scope,
                           **kwargs)
    if loss is not None:
        losses.append(loss)
        grads_and_vars.append(grad)

    # Compute the total_loss summing all the losses.
    total_loss = tf.add_n(losses, name='total_loss')
    # Sum the gradients across clones.
    grads_and_vars = _gradients(grads_and_vars)

    return total_loss, grads_and_vars


def rotate_around_point(angle,data,point):
    """
    :param angle: rotation angele
    :param data: point
    :param point: rotation center point
    :return:
    """
    rotate_x = (data[:, 0] - point[0])*np.cos(angle) - (data[:, 1] - point[1])*np.sin(angle) + point[0]
    rotate_y = (data[:, 0] - point[0])*np.sin(angle) + (data[:, 1] - point[1])*np.cos(angle) + point[1]
    rotate_z = data[:, 2]
    return np.c_[rotate_x, rotate_y, rotate_z]


def rotate_around_point_x(angle, data, point):
    """
    :param angle: rotation angle
    :param data: point
    :param point: rotation center point
    :return:
    """
    rotate_x = data[:, 0]
    rotate_y = (data[:, 1] - point[1])*np.cos(angle) - (data[:, 2] - point[2])*np.sin(angle) + point[1]
    rotate_z = (data[:, 1] - point[1])*np.sin(angle) + (data[:, 2] - point[2])*np.cos(angle) + point[2]
    return np.c_[rotate_x, rotate_y, rotate_z]


def rotate_around_point_y(angle, data, point):
    """
    :param angle: rotation angle
    :param data: point
    :param point: rotation center point
    :return:
    """
    rotate_x = (data[:, 2] - point[2])*np.sin(angle) + (data[:, 0] - point[0])*np.cos(angle) + point[0]
    rotate_y = data[:, 1]
    rotate_z = (data[:, 2] - point[2])*np.cos(angle) - (data[:, 0] - point[0])*np.sin(angle) + point[2]
    return np.c_[rotate_x, rotate_y, rotate_z]


def get_profile_data(input_data, grid_x, grid_z, number, char):
    """
    :param input_data:
    :param grid_x:
    :param grid_z:
    :param number:
    :param char:
    :return:
    """
    # rotate_nums = int(360 / angle)
    # angle_nD = 360 / number
    profile_vector = np.zeros((1, number*grid_x*grid_z))
    points_pixel_num_zx = []
    pts1 = 0
    # for i in range(rotate_nums):
    num_profile_vector = 0

    for i_1 in range(number):
        if i_1 == 0:
            # input_data1 = input_data
            pts1 += input_data.shape[0]
        max_x = np.max(input_data[:, 0])
        min_x = np.min(input_data[:, 0])

        max_z = np.max(input_data[:, 2])
        min_z = np.min(input_data[:, 2])

        deta_x = max_x - min_x
        deta_z = max_z - min_z

        deta_deta_xz = np.abs(deta_x - deta_z)/2
        for j in range(pts1):
            point = input_data[j,:]
            if (deta_x > deta_z):
                if (j == 0):
                    pedeta_x = deta_x/grid_x
                    pedeta_z = deta_x/grid_z
                    attachment_z = np.ceil(deta_deta_xz/pedeta_z)
                x_num = np.ceil((point[0]-min_x)/pedeta_x)
                z_num = (np.ceil((point[2] - min_z) / pedeta_z) + attachment_z)
                if (x_num == 0):
                    x_num = 1
                if (z_num == 0):
                    z_num = 1
                z_num = (grid_z + 1) - z_num
            else:
                if(j == 0):
                    pedeta_x = deta_z / grid_x
                    pedeta_z = deta_z / grid_z
                    attachment_x = np.ceil(deta_deta_xz / pedeta_x)
                x_num = (np.ceil((point[0] - min_x) / pedeta_x) + attachment_x)
                z_num = np.ceil((point[2] - min_z) / pedeta_z)
                if (x_num == 0):
                    x_num = 1
                if (z_num == 0):
                    z_num = 1
                z_num = (grid_z + 1) - z_num
            points_pixel_num_zx.append([z_num, x_num])
        points_pixel_num_zx = np.array(points_pixel_num_zx)

        matrix_value_y = np.zeros((grid_z,grid_x))
        bar = tqdm(range(grid_z))
        for k in bar:
            bar.set_description("Processing %s" % char)
            for h in range(grid_x):
                n_z = [in_z for in_z,z_ in enumerate(points_pixel_num_zx[:, 0]) if z_ == (k+1)]
                n_x = [in_x for in_x,x_ in enumerate(points_pixel_num_zx[:, 1]) if x_ == (h+1)]
                grid_ij_points_num_zx = list(set(n_z).intersection(set(n_x)))

                if grid_ij_points_num_zx != []:
                    matrix_value_y[k,h] = 1
                profile_vector[0,num_profile_vector] = matrix_value_y[k,h]
                num_profile_vector +=1

    return np.array(profile_vector)


def get_xoy_profile_data(index_1, index_2, input_data, grid_x, grid_y):
    """
    :param input_data:
    :param grid_x:
    :param grid_y:
    :param number:
    :param char:
    :return:
    """
    # rotate_nums = int(360 / angle)
    # angle_nD = 360 / number
    number = 1
    profile_vector = np.zeros((1, number*grid_x*grid_y))
    points_pixel_num_yx = []
    pts1 = 0
    # for i in range(rotate_nums):
    num_profile_vector = 0

    for i_1 in range(number):
        if i_1 == 0:
            # input_data1 = input_data
            pts1 += input_data.shape[0]
        max_x = np.max(input_data[:, 0])
        min_x = np.min(input_data[:, 0])

        max_y = np.max(input_data[:, 1])
        min_y = np.min(input_data[:, 1])

        deta_x = max_x - min_x
        deta_y = max_y - min_y

        deta_deta_xy = np.abs(deta_x - deta_y)/2
        for j in range(pts1):
            point = input_data[j, :]
            if deta_x > deta_y:
                if j == 0:
                    pedeta_x = deta_x/grid_x
                    pedeta_y = deta_x/grid_y
                    attachment_y = np.ceil(deta_deta_xy/pedeta_y)
                x_num = np.ceil((point[0]-min_x)/pedeta_x)
                y_num = (np.ceil((point[1] - min_y) / pedeta_y) + attachment_y)
                if x_num == 0:
                    x_num = 1
                if y_num == 0:
                    y_num = 1
                y_num = (grid_y + 1) - y_num
            else:
                if j == 0:
                    pedeta_x = deta_y / grid_x
                    pedeta_y = deta_y / grid_y
                    attachment_x = np.ceil(deta_deta_xy / pedeta_x)
                x_num = (np.ceil((point[0] - min_x) / pedeta_x) + attachment_x)
                y_num = np.ceil((point[1] - min_y) / pedeta_y)
                if (x_num == 0):
                    x_num = 1
                if (y_num == 0):
                    y_num = 1
                y_num = (grid_y + 1) - y_num
            points_pixel_num_yx.append([y_num, x_num])
        points_pixel_num_yx = np.array(points_pixel_num_yx)

        matrix_value_y = np.zeros((grid_y,grid_x))
        bar = tqdm(range(grid_y))
        for k in bar:
            bar.set_description("Processing %d of current batch, index %d" % (index_1, index_2))
            for h in range(grid_x):
                n_y = [in_y for in_y,y_ in enumerate(points_pixel_num_yx[:, 0]) if y_ == (k+1)]
                n_x = [in_x for in_x,x_ in enumerate(points_pixel_num_yx[:, 1]) if x_ == (h+1)]
                grid_ij_points_num_yx = list(set(n_y).intersection(set(n_x)))

                if grid_ij_points_num_yx:
                    matrix_value_y[k, h] = 1
                profile_vector[0, num_profile_vector] = matrix_value_y[k, h]
                num_profile_vector += 1

    return np.array(profile_vector)


def pointcloud_multiview_generate(index_1, data_curr, grid_x, grid_z, angle):
    angle_ = angle * (np.pi / 180)
    local_ori = (np.max(data_curr, axis=0) - np.min(data_curr, axis=0)) / 2 + np.min(data_curr, axis=0)
    center_point = local_ori
    multi_view_array = []
    for i in range(int(360 / angle)):
        rotate_angle_ = i * angle_
        rotated_data = rotate_around_point_y(rotate_angle_, data_curr, center_point)

        profile_xoz1 = np.array(get_xoy_profile_data(index_1, i, rotated_data, grid_x, grid_z)).reshape((1, -1))

        Image_r = profile_xoz1.reshape(-1, grid_z)

        nor_image_color_gray = image_color_gray*(1. / 255) - 0.5
        nor_image_color_white = image_color_white*(1. / 255) - 0.5

        rgbArray = np.zeros((grid_x, grid_z, 3))
        rgbArray[..., 0] = Image_r * nor_image_color_gray
        index_0 = (rgbArray[..., 0] == 0)
        rgbArray[index_0, 0] = nor_image_color_white
        rgbArray[..., 1] = Image_r * nor_image_color_gray
        rgbArray[index_0, 1] = nor_image_color_white
        rgbArray[..., 2] = Image_r * nor_image_color_gray
        rgbArray[index_0, 2] = nor_image_color_white

        multi_view_array.append(rgbArray)
    return multi_view_array


def mini_batch_pointcloud_multiview_generate(batch_data, im_width, im_height, rotate_angle):
    batch_size = batch_data.shape[0]
    batch_data_multi_view = []
    for i in range(batch_size):
        current_pointcloud = batch_data[i]
        current_multi_view = pointcloud_multiview_generate(i, current_pointcloud, im_width, im_height, rotate_angle)
        batch_data_multi_view.append(current_multi_view)
    return batch_data_multi_view


def fast_confusion(true, pred, label_values=None):

   """
   Fast confusion matrix (100x faster than Scikit learn). But only works if labels are la
   :param true:
   :param false:
   :param num_classes:
   :return:
   """

   true = np.squeeze(true)
   pred = np.squeeze(pred)
   if len(true.shape) != 1:
       raise ValueError('Truth values are stored in a {:d}D array instead of 1D array'. format(len(true.shape)))
   if len(pred.shape) != 1:
       raise ValueError('Prediction values are stored in a {:d}D array instead of 1D array'. format(len(pred.shape)))
   if true.dtype not in [np.int32, np.int64]:
       raise ValueError('Truth values are {:s} instead of int32 or int64'.format(true.dtype))
   if pred.dtype not in [np.int32, np.int64]:
       raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(pred.dtype))
   true = true.astype(np.int32)
   pred = pred.astype(np.int32)

   if label_values is None:
       label_values = np.unique(np.hstack((true, pred)))
   else:
       if label_values.dtype not in [np.int32, np.int64]:
           raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
       if len(np.unique(label_values)) < len(label_values):
           raise ValueError('Given labels are not unique')

   label_values = np.sort(label_values)
   num_classes = len(label_values)
   if label_values[0] == 0 and label_values[-1] == num_classes - 1:
       vec_conf = np.bincount(true * num_classes + pred)

       if vec_conf.shape[0] < num_classes ** 2:
           vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')
       return vec_conf.reshape((num_classes, num_classes))


   else:
       if label_values[0] < 0:
           raise ValueError('Unsupported negative classes')

       label_map = np.zeros((label_values[-1] + 1,), dtype=np.int32)
       for k, v in enumerate(label_values):
           label_map[v] = k

       pred = label_map[pred]
       true = label_map[true]

       vec_conf = np.bincount(true * num_classes + pred)

       # Add possible missing values due to classes not being in pred or true
       if vec_conf.shape[0] < num_classes ** 2:
           vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')

       # Reshape confusion in a matrix
       return vec_conf.reshape((num_classes, num_classes))


if __name__ == '__main__':
    start = time.time()

    data_path = './data/train_files.txt'
    TRAIN_FILES = getDataFiles(data_path)
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    for fn in range(len(TRAIN_FILES)):
        current_data, current_label = loadDataFile(TRAIN_FILES[train_file_idxs[fn]])

        file_size = current_data.shape[0]
        num_batches = file_size // 2

        for batch_idx in range(num_batches):
            start_idx = batch_idx * 2
            end_idx = (batch_idx+1) * 2

            current_batch_train_data = current_data[start_idx:end_idx, :, :]
            current_batch_data_label = current_label[start_idx:end_idx]

            current_train_multi_views = mini_batch_pointcloud_multiview_generate(current_batch_train_data, 299, 299, 360)
            current_train_multi_views = np.array(current_train_multi_views)
            print(current_train_multi_views.shape)

    t = (time.time() - start)
    print("running time:{:.2f} s\n".format(t))