try:
    from matplotlib import use
    use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def draw_losses_test_data(nnpu_test, nnpusb_test):
    plots = []
    legend = []
    title = 'MNIST:Test Error Rate'

    nnpn_test_plot, = plt.plot(nnpu_test, 'r-')
    nnpusb_test_plot, = plt.plot(nnpusb_test, 'b-')
    plots.extend([nnpn_test_plot, nnpusb_test_plot])
    legend.extend(['nnPU test', 'nnPUSB test'])

    plt.legend(
        plots,
        legend,
        loc='upper right'
    )
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.grid(True, linestyle="-.")
    plt.savefig("./result/error_rate.png")
    plt.show()


def draw_precision_recall(nnpu_precision, nnpu_recall, nnpusb_precision, nnpusb_recall):
    plots = []
    legend = []
    title = 'MNIST:Precision and Recall'

    nnpn_precision_plot, = plt.plot(nnpu_precision, 'r-')
    nnpn_recall_plot, = plt.plot(nnpu_recall, 'g--')

    nnpusb_precision_plot, = plt.plot(nnpusb_precision, 'b-')
    nnpusb_recall_plot, = plt.plot(nnpusb_recall, 'y--')

    plots.extend([nnpn_precision_plot, nnpn_recall_plot,
                  nnpusb_precision_plot, nnpusb_recall_plot])
    legend.extend(['nnPU: Precision', 'nnPU: Recall',
                   'nnPUSB: Precision', 'nnPUSB: Recall'])

    plt.legend(
        plots,
        legend,
        loc='upper right'
    )
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True, linestyle="-.")
    plt.savefig("./result/precision_recall.png")
    plt.show()
