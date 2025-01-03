import keras.backend as K
import tensorflow as tf


def loss_seg(relative_weights): #relative_weights:  chứa các trọng số của với 3 lớp (Foreground, Background, Border)
    """
    Calculates Cross-Entropy Loss between the class targets and predicted outputs.
    Predicted outputs consist of three classes: Foreground, Background and Border.
    Class predictions are weighted by the parameter `relative_weights`.
    """

    class_weights = tf.constant([relative_weights]) #Tạo môt tensor từ relative_weights và chuyển nó thành hằng số TensorFlow.

    def seg_crossentropy(class_targets, y_pred):#class_targets: Mảng mục tiêu (ground truth) dưới dạng một tensor. Thường là các nhãn phân đoạn cho hình ảnh đầu vào, y_pred: Dự đoán của mô hình, có thể là các giá trị logits (trước khi áp dụng hàm softmax).
        onehot_labels = tf.reshape(class_targets, [-1, 3]) #Chuyển đổi class_targets thành định dạng one-hot, với 3 lớp (Foreground, Background, Border), mỗi pixel sẽ có một vector one-hot với chiều dài 3.
        #Tính toán trọng số cho từng pixel dựa trên các lớp mục tiêu one-hot. Mỗi pixel sẽ có trọng số được tính bằng cách nhân các trọng số lớp với các giá trị one-hot tương ứng, và sau đó tính tổng theo trục axis=1.
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)

        a = tf.reduce_sum(onehot_labels, axis=-1) #Tính tổng các giá trị trong vector one-hot của mỗi pixel.

        #softmax_cross_entropy_with_logits_v2: Đây là hàm tính Cross-Entropy Loss cho bài toán phân loại đa lớp.
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                          logits=tf.reshape(y_pred, [-1, 3])) #Hàm này sẽ tính toán loss cho từng lớp của mỗi pixel.

        weighted_loss = loss * weights #Tính loss có trọng số

        return K.mean(a * weighted_loss) # Tính toán giá trị trung bình của weighted loss. Mỗi pixel có thể có trọng số khác nhau, vì vậy việc tính trung bình này giúp mô hình tập trung vào các phần quan trọng của ảnh (như Border hoặc Foreground).
    return seg_crossentropy #Hàm này sẽ được trả về bởi loss_seg để có thể được sử dụng như là hàm mất mát trong quá trình huấn luyện mô hình.

"""Hàm loss_seg là một hàm tính Cross-Entropy Loss cho bài toán phân đoạn, với ba lớp (Foreground, Background, Border).
 Nó sử dụng trọng số lớp (relative_weights) để điều chỉnh sự quan trọng của mỗi lớp trong quá trình tính toán loss. 
 Các bước chính trong hàm là chuyển nhãn thành one-hot encoding, tính toán loss theo softmax, áp dụng trọng số cho loss và tính trung bình các giá trị loss.
 Hàm này sau đó trả về hàm seg_crossentropy có thể được sử dụng trong quá trình huấn luyện mô hình."""