class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure()
        self.plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.
        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.
        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc = "upper right")

    def roc_plot(self, fpr, tpr):
        """
        Plotting the ROC Curve. 
        fpr : list [ float ]
            false positive rate
        tpr : list [ float ]
            true positive rate
        return : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(fpr, tpr)
        ax.plot([0,1], [0,1], color = 'black', linestyle = 'dashed')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.legend(["ROC Curve", "Chance Level (AUC = 0.5)"], loc = "lower right")

    def save_figure(self, name):
        """
        Save figure.
        name : str
            save .png file path.
        return : None
        """
        self.plt.savefig(name)