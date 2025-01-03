import keras.backend as K
import tensorflow as tf


def loss_seg(relative_weights): #relative_weights:  ch?a các tr?ng s? týõng ð?i c?a ba l?p (Foreground, Background, Border)
    """
    Calculates Cross-Entropy Loss between the class targets and predicted outputs.
    Predicted outputs consist of three classes: Foreground, Background and Border.
    Class predictions are weighted by the parameter `relative_weights`.
    """

    class_weights = tf.constant([relative_weights]) #T?o m?t tensor t? relative_weights và chuy?n nó thành h?ng s? TensorFlow.

    def seg_crossentropy(class_targets, y_pred):#class_targets:  M?ng m?c tiêu (ground truth) dý?i d?ng m?t tensor Thý?ng là các nh?n phân ðo?n cho h?nh ?nh ð?u vào , y_pred: D? ðoán c?a mô h?nh, có th? là các giá tr? logits 
        onehot_labels = tf.reshape(class_targets, [-1, 3]) #Chuy?n ð?i class_targets thành ð?nh d?ng one-hot, v?i 3 l?p (Foreground, Background, Border), m?i pixel s? có m?t vector one-hot v?i chi?u dài 3.
        #Tính toán tr?ng s? cho t?ng pixel d?a trên các l?p m?c tiêu one-hot. M?i pixel s? có tr?ng s? ðý?c tính b?ng cách nhân các tr?ng s? l?p v?i các giá tr? one-hot týõng ?ng
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)

        a = tf.reduce_sum(onehot_labels, axis=-1) #Tính t?ng các giá tr? trong vector one-hot c?a m?i pixeL

        #softmax_cross_entropy_with_logits_v2: Ðây là hàm tính Cross-Entropy Loss cho bài toán phân lo?i ða l?p
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                          logits=tf.reshape(y_pred, [-1, 3])) #Hàm này s? tính toán loss cho t?ng l?p c?a m?i pixel.

        weighted_loss = loss * weights #Tính loss có tr?ng s?

        return K.mean(a * weighted_loss) #Tính toán giá tr? trung b?nh c?a weighted loss. M?i pixel có th? có tr?ng s? khác nhau, v? v?y vi?c tính trung b?nh này giúp mô h?nh t?p trung vào các ph?n quan tr?ng c?a ?nh (nhý Border ho?c Foreground).

    return seg_crossentropy #Hàm này s? ðý?c tr? v? b?i loss_seg ð? có th? ðý?c s? d?ng nhý là hàm m?t mát trong quá tr?nh hu?n luy?n mô h?nh.

"""Hàm loss_seg là m?t hàm tính Cross-Entropy Loss cho bài toán phân ðo?n, v?i ba l?p (Foreground, Background, Border). 
Nó s? d?ng tr?ng s? l?p (relative_weights) ð? ði?u ch?nh s? quan tr?ng c?a m?i l?p trong quá tr?nh tính toán loss. 
Các bý?c chính trong hàm là chuy?n nh?n thành one-hot encoding, 
tính toán loss theo softmax, 
áp d?ng tr?ng s? cho loss và tính trung b?nh các giá tr? loss. 
Hàm này sau ðó tr? v? hàm seg_crossentropy có th? ðý?c s? d?ng trong quá tr?nh hu?n luy?n mô h?nh."""