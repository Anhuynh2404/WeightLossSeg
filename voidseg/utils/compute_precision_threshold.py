import numpy as np
from numba import jit
from scipy import ndimage
from tqdm import tqdm, tqdm_notebook

@jit
def pixel_sharing_bipartite(lab1, lab2): #(ground truth với dự đoán).
    assert lab1.shape == lab2.shape #Hàm này kiểm tra xem hai mảng lab1 và lab2 có kích thước giống nhau hay không
    psg = np.zeros((lab1.max() + 1, lab2.max() + 1), dtype=np.int) #Ma trận vuông psg sẽ lưu trữ số lần mà mỗi cặp nhãn trong lab1 và lab2 chia sẻ các pixel.
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1 #Tại mỗi vị trí pixel, giá trị tương ứng trong ma trận psg sẽ được tăng lên.
    return psg

#IoU là một chỉ số phổ biến trong phân đoạn hình ảnh để đo lường độ chính xác của dự đoán.
def intersection_over_union(psg):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    rsum = np.sum(psg, 0, keepdims=True) #  Tính tổng theo các cột của ma trận psg:
    csum = np.sum(psg, 1, keepdims=True) # Tính tổng theo các hàng của ma trận psg:
    return psg / (rsum + csum - psg)
"""
IoU(A,B)= 
∣A∪B∣
∣A∩B∣
|A ∩ B| là phần giao nhau giữa hai tập A và B, được đại diện bằng các giá trị trong ma trận psg.
|A ∪ B| là phần hợp của hai tập A và B.
"""

#hàm này thực hiện việc xác định xem các nhãn có chồng lắp với nhau (dựa trên một ngưỡng IoU) hay không.
def matching_iou(psg, fraction=0.5):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    iou = intersection_over_union(psg)
    matching = iou > 0.5 # một ngưỡng được áp dụng để xác định các cặp nhãn có sự chồng lắp. Cụ thể, nếu IoU giữa hai lớp nhãn lớn hơn 0.5, thì chúng được coi là matching.
    #Dòng lệnh này đảm bảo rằng không có sự phù hợp nào giữa nhãn nền (thường được chỉ định là 0) với bất kỳ nhãn nào khác, vì nhãn nền không quan tâm đến việc đánh giá sự phù hợp trong phân đoạn.
    matching[:, 0] = False
    matching[0, :] = False
    return matching


#tính toán precision (độ chính xác) trong bài toán phân đoạn hình ảnh bằng cách sử dụng chỉ số Intersection over Union (IoU) cho các lớp nhãn
def precision(lab_gt, lab, iou=0.5, partial_dataset=False):
    """
    precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching
    :Authors:
        Coleman Broaddus
    """
    psg = pixel_sharing_bipartite(lab_gt, lab) #để tạo ra ma trận pixel sharing bipartite (psg), đại diện cho sự phân bổ pixel giữa nhãn thực (lab_gt) và nhãn dự đoán (lab). Ma trận này có kích thước là (n_classes_gt, n_classes_pred) (tương ứng với các nhãn trong lab_gt và lab).
    matching = matching_iou(psg, fraction=iou)
    # Đảm bảo rằng không có nhãn nào được ghép đôi nhiều lần:
    assert matching.sum(0).max() < 2
    assert matching.sum(1).max() < 2
    #Tính toán số lượng nhãn trong lab_gt (nhãn thực) và lab (nhãn dự đoán):
    n_gt = len(set(np.unique(lab_gt)) - {0})
    n_hyp = len(set(np.unique(lab)) - {0})
    #Tính toán số lượng nhãn đã được ghép đôi (matched):
    n_matched = matching.sum()
    #Nếu partial_dataset=True, hàm trả về số lượng nhãn đã ghép đôi và số nhãn chưa ghép đôi.
    if partial_dataset:
        return n_matched, (n_gt + n_hyp - n_matched)
    else:
        # hàm trả về độ chính xác (precision)
        return n_matched / (n_gt + n_hyp - n_matched)


#dùng để kiểm tra xem mã nguồn đang được thực thi trong môi trường Jupyter Notebook hay không 
def isnotebook():
    """
    Checks if code is run in a notebook, which can be useful to determine what sort of progressbar to use.
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/24937408#24937408

    Returns
    -------
    bool
        True if running in notebook else False.

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


#được sử dụng để tính toán Precision (độ chính xác) tại các ngưỡng xác suất khác nhau trên dữ liệu kiểm tra và trả về ngưỡng tốt nhất, tức là ngưỡng cho phép mô hình đạt độ chính xác cao nhất.
"""
X_val: Mảng chứa các ảnh xác nhận (validation images).
Y_val: Mảng chứa các nhãn xác nhận (validation labels).
model: Mô hình học sâu (Keras model) để dự đoán nhãn cho ảnh.
mode: Một tham số tùy chọn, có thể là 'none' hoặc 'StarDist'. Nếu là 'StarDist', mô hình StarDist sẽ được sử dụng, nếu không, mô hình U-net sẽ được sử dụng.
"""
def compute_threshold(X_val, Y_val, model, mode=None):
    """
    Computes average precision (AP) at different probability thresholds on validation data and returns the best-performing threshold.

    Parameters
    ----------
    X_val : array(float)
        Array of validation images.
    Y_val : array(float)
        Array of validation labels
    model: keras model

    mode: 'none', 'StarDist'
        If `none`, consider a U-net type model, else, considers a `StarDist` type model
    Returns
    -------
    computed_threshold: float
        Best-performing threshold that gives the highest AP.


    """
    print('Computing best threshold: ')
    precision_scores = [] #Để lưu trữ các điểm Precision tại các ngưỡng khác nhau.
    #Hàm kiểm tra xem mã có chạy trong Jupyter Notebook hay không để chọn kiểu tiến trình phù hợp.
    if (isnotebook()):
        progress_bar = tqdm_notebook
    else:
        progress_bar = tqdm
    for ts in progress_bar(np.linspace(0.1, 1, 19)): #Hàm sẽ thử các ngưỡng từ 0.1 đến 1, chia thành 19 mức khác nhau với np.linspace(0.1, 1, 19).
        precision_score = 0
        #Đối với mỗi ngưỡng xác suất (ts), hàm lặp qua tất cả các ảnh trong tập dữ liệu xác nhận (X_val và Y_val), tính toán độ chính xác (precision) cho từng ảnh.
        for idx in range(X_val.shape[0]):
            img, gt = X_val[idx], Y_val[idx]
            if (mode == "StarDist"):                
                labels, _ = model.predict_instances(img, prob_thresh=ts) #Nếu mode là "StarDist", sử dụng phương thức predict_instances của mô hình để dự đoán các nhãn (labels) cho ảnh, với ngưỡng xác suất ts.
            else:#Nếu không phải StarDist (tức là mô hình U-net):
                prediction = model.predict(img, axes='YX')
                prediction_exp = np.exp(prediction[..., :])
                prediction_precision = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
                prediction_fg = prediction_precision[..., 1]
                pred_thresholded = prediction_fg > ts
                labels, _ = ndimage.label(pred_thresholded)

            tmp_score = precision(gt, labels)
            if not np.isnan(tmp_score):
                precision_score += tmp_score

        precision_score /= float(X_val.shape[0])
        precision_scores.append((ts, precision_score))
        print('Precision-Score for threshold =', "{:.2f}".format(ts), 'is', "{:.4f}".format(precision_score))

    best_score = sorted(precision_scores, key=lambda tup: tup[1])[-1]
    computed_threshold = best_score[0]
    return computed_threshold
