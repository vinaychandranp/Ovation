import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
def xrange(x):

    return iter(range(x))


def plot_cm(cm, values, title='Confusion matrix', cmap=plt.cm.rainbow):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cb = fig.colorbar(res)
    tick_marks = np.arange(len(values))
    plt.xticks(tick_marks, values)
    plt.yticks(tick_marks, values)
    fig.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    width, height = cm.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

def compute_confusion_matrix(y_test, y_pred,fname, plot=True):

    #update index

    y_test = np.array(y_test) + 1
    y_pred = np.array(y_pred) + 1

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    if plot:
        values = np.unique(y_pred)

        plot_confusion_matrix(cm, values,fname)
        #plot_cm(cm, values)
        plt.show()
    return cm



def plot_confusion_matrix(cm, classes,fname,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,V = 100):

    import itertools
    from matplotlib.font_manager import FontProperties
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:

        cm = cm.astype('float')
        for j in range(cm.shape[0]):
            cm[j,:]/=np.sum(cm[j,:])
        cm=cm*V
        print("Normalized confusion matrix")
        print(cm)



    #for j in range(cm.shape[0]):
    #    print(np.sum(cm[j, :]))


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    font0 = FontProperties()

    font1 = font0.copy()

    font1.set_style('italic')
    font1.set_weight('bold')
    font1.set_size('xx-large')

    font0.set_size('x-large')


    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(j, i, "{0:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                     color="yellow" if i == j  else "black",fontproperties=font1 if i == j else font0)

    plt.tight_layout()
    plt.ylabel('True records')
    plt.xlabel('Predicted records')
    plt.savefig(fname)
    #plt.show()



def read_txt_file_classification(fname):
    dataset = []
    tot = 0
    with open(fname,'r') as f_res:
        lines = f_res.read().splitlines()
        for line in lines:
            text, pred, target = line.split('\t')
            tot+=1
            y_p = ast.literal_eval(pred)
            y_t = ast.literal_eval(target)


            class_predicted = np.argmax(y_p)
            class_target = np.argmax(y_t)
            #print(y_p, y_t, class_predicted, class_target)
            dataset.append((text, y_p, y_t, class_predicted, class_target))

        print('TOT LINES', tot)

    return dataset

def read_txt_file_regression(fname):
    dataset = []
    tot = 0
    with open(fname,'r') as f_res:
        lines = f_res.read().splitlines()
        for line in lines:
            text, pred, target = line.split('\t')
            tot+=1
            y_p = float(pred)
            y_t = float(target)


            class_predicted = int(np.floor(y_p + 0.5))
            class_target = int(y_t)
            #print(y_p, y_t, class_predicted, class_target)
            dataset.append((text, y_p, y_t, class_predicted, class_target))

        print('TOT LINES', tot)

    return dataset


def compute_measure_classification(fname, ext='pdf'):
    print('READ DATA', fname)
    a, b = os.path.split(fname)
    plot_fname = os.path.join(a,b.split('.')[0]+'_plot.'+ext)
    dataset = read_txt_file_classification(fname)
    _, _, _, y_pred, y_target = zip(*dataset)

    print('Evaluation Metrics')

    print('ACCURACY', accuracy_score(y_pred, y_target))
    labels = range(max(y_target)+1)
    precision, recall, fbeta, support = precision_recall_fscore_support(y_target, y_pred, average=None,labels=labels)

    print('Precision, Recall, F1')
    print(';'.join(map(str,labels)))
    print(np.array([precision,recall,fbeta]))
    compute_confusion_matrix(y_target, y_pred, plot_fname,plot=True)


def compute_measure_regression(fname, ext='pdf'):
    print('READ DATA', fname)
    a, b = os.path.split(fname)
    plot_fname = os.path.join(a,b.split('.')[0]+'_plot.'+ext)
    dataset = read_txt_file_regression(fname)
    _, _, _, y_pred, y_target = zip(*dataset)

    print('Evaluation Metrics')

    print('ACCURACY', accuracy_score(y_pred, y_target))
    labels = range(max(y_target)+1)
    precision, recall, fbeta, support = precision_recall_fscore_support(y_target, y_pred, average=None,labels=labels)

    print('Precision, Recall, F1')
    print(';'.join(map(str,labels)))
    print(np.array([precision,recall,fbeta]))
    compute_confusion_matrix(y_target, y_pred, plot_fname,plot=True)


#compute_measure_classification('/home/scstech/WORK/ovation_proj/Ovation/test_samples_2363.txt')
