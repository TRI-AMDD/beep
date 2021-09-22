# Predict


![cli_predict](../static/op_graphic_predict.png)


## Predict help dialog


```shell
$: beep predict --help

Usage: beep predict [OPTIONS] MODEL_FILE

  Run a previously trained model to predict degradation targets.The MODEL_FILE
  passed should be an output of 'beep train' or aserialized
  BEEPLinearModelExperiment object.

Options:
  -fm, --feature-matrix-file TEXT
                                  Feature matrix to use as input to the model.
                                  Predictions are basedon these features.
                                  [required]
  -o, --output-filename FILE      Filename (json) to write the final predicted
                                  dataframe to.
  --predict-sample-nan-thresh FLOAT
                                  Threshold to keep a sample from any
                                  prediction set.
  --help                          Show this message and exit.

```