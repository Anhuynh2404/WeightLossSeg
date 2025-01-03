import keras.backend as K
import tensorflow as tf


def loss_seg(relative_weights): #relative_weights:  ch?a c�c tr?ng s? t��ng �?i c?a ba l?p (Foreground, Background, Border)
    """
    Calculates Cross-Entropy Loss between the class targets and predicted outputs.
    Predicted outputs consist of three classes: Foreground, Background and Border.
    Class predictions are weighted by the parameter `relative_weights`.
    """

    class_weights = tf.constant([relative_weights]) #T?o m?t tensor t? relative_weights v� chuy?n n� th�nh h?ng s? TensorFlow.

    def seg_crossentropy(class_targets, y_pred):#class_targets:  M?ng m?c ti�u (ground truth) d�?i d?ng m?t tensor Th�?ng l� c�c nh?n ph�n �o?n cho h?nh ?nh �?u v�o , y_pred: D? �o�n c?a m� h?nh, c� th? l� c�c gi� tr? logits 
        onehot_labels = tf.reshape(class_targets, [-1, 3]) #Chuy?n �?i class_targets th�nh �?nh d?ng one-hot, v?i 3 l?p (Foreground, Background, Border), m?i pixel s? c� m?t vector one-hot v?i chi?u d�i 3.
        #T�nh to�n tr?ng s? cho t?ng pixel d?a tr�n c�c l?p m?c ti�u one-hot. M?i pixel s? c� tr?ng s? ��?c t�nh b?ng c�ch nh�n c�c tr?ng s? l?p v?i c�c gi� tr? one-hot t��ng ?ng
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)

        a = tf.reduce_sum(onehot_labels, axis=-1) #T�nh t?ng c�c gi� tr? trong vector one-hot c?a m?i pixeL

        #softmax_cross_entropy_with_logits_v2: ��y l� h�m t�nh Cross-Entropy Loss cho b�i to�n ph�n lo?i �a l?p
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                          logits=tf.reshape(y_pred, [-1, 3])) #H�m n�y s? t�nh to�n loss cho t?ng l?p c?a m?i pixel.

        weighted_loss = loss * weights #T�nh loss c� tr?ng s?

        return K.mean(a * weighted_loss) #T�nh to�n gi� tr? trung b?nh c?a weighted loss. M?i pixel c� th? c� tr?ng s? kh�c nhau, v? v?y vi?c t�nh trung b?nh n�y gi�p m� h?nh t?p trung v�o c�c ph?n quan tr?ng c?a ?nh (nh� Border ho?c Foreground).

    return seg_crossentropy #H�m n�y s? ��?c tr? v? b?i loss_seg �? c� th? ��?c s? d?ng nh� l� h�m m?t m�t trong qu� tr?nh hu?n luy?n m� h?nh.

"""H�m loss_seg l� m?t h�m t�nh Cross-Entropy Loss cho b�i to�n ph�n �o?n, v?i ba l?p (Foreground, Background, Border). 
N� s? d?ng tr?ng s? l?p (relative_weights) �? �i?u ch?nh s? quan tr?ng c?a m?i l?p trong qu� tr?nh t�nh to�n loss. 
C�c b�?c ch�nh trong h�m l� chuy?n nh?n th�nh one-hot encoding, 
t�nh to�n loss theo softmax, 
�p d?ng tr?ng s? cho loss v� t�nh trung b?nh c�c gi� tr? loss. 
H�m n�y sau �� tr? v? h�m seg_crossentropy c� th? ��?c s? d?ng trong qu� tr?nh hu?n luy?n m� h?nh."""