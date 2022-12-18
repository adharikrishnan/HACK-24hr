
# For paths and any helper functions
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


# Dataset path
DATASET_PATH = 'Dataset/'

# Helper function to create Confusion Matrix, code for reference 
def display_cm(CM, classes, title = "Confusion Matrix", cmap = plot.cm.Blues):
    plot.imshow(CM, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()

    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    fmt = 'd'
    threshold = CM.max()

    for x, y in itertools.product(range(CM.shape[0]), range(CM.shape[1])):
        plot.text(y,x, format(CM[x, y],fmt), horizontalalignment = 'left')
        color = 'red' if CM[x,y] > threshold else 'black'

    
    plot.tight_layout()
    plot.ylabel('Actual Value')
    plot.xlabel('Predicted Value')
    
