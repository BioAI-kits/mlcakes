## mlcakes

Some functional modules for machine learning.

## file structure

```
|- evaluation
    |- binary_class.py # metics for binary class task.
```

## API

###  `evaluation.binary_class.Metrics`

**Parameters:**

-  y_pred

> Model predicted labels [list or numpy.array]. such as: [1,0,1,0,1,0]

- y_true

> Real labels [list or numpy.array]. such as: [1,1,1,0,1,0]

- y_score

> The binary classification model predicts the probability value that a sample is positive. such as [0.1, 0.8, 0.7, 0.6, 0.3, 0.2]

- output

> JSON file name, used to save all metrics. such as model_metrics.json

- plot_roc

> Bool value, whether to draw the ROC curve. such as True or False.

- roc_fig

> Filename, used to save the ROC curve. such as roc_curve.pdf, roc_curve.png

**Example**

```py
import numpy as np
from evaluation.binary_class import Metrics

y_pred = np.random.choice(2, size=100)
y_true = np.random.choice(2, size=100)
y_score = np.random.random(100)
    
Evaluation = Metrics(y_pred=y_pred, y_score=y_score, y_true=y_true, plot_roc=True ,roc_fig='aa.png')
print(Evaluation.metrics_)
```

