# PP Test 1 sec Ensemble

This notebook has the final code used to create the ringed seal detector. 


```python
import pandas as pd
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
from ketos.audio.spectrogram import MagSpectrogram
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold
import numpy as np
import tensorflow as tf
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.neural_networks.resnet import ResNetInterface
import shutil
from ketos.data_handling.data_feeding import JointBatchGen
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import csv
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn

print('done')
```

    C:\Users\kzammit\Miniconda3\envs\ketos_env\lib\site-packages\keras\optimizer_v2\adam.py:105: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(Adam, self).__init__(name, **kwargs)
    

    done
    

## User Inputs


```python
main_folder = r'C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test'
spec_file = r'C:\Users\kzammit\Documents\Detector\detector-1sec\inputs\spec_config_1sec.json'
data_folder = r'D:\ringed-seal-data'
db_name = main_folder + '\\' r'pierce_point_db.h5'
```

## Step One: Create Database 

A database consisting of manually verified spectrogram segments is created using excel workbooks.


```python
## Create Database ##

pp_pos = pd.read_excel(main_folder + '\\' + 'std_PP_positives.xlsx')
pp_pos2 = pp_pos.ffill()
pp_pos2 = sl.standardize(table=pp_pos2, start_labels_at_1=True)
print('Positives standardized? ' + str(sl.is_standardized(pp_pos2)))

pp_neg = pd.read_excel(main_folder + '\\' + 'std_PP_negatives-manual-FINAL.xlsx')
pp_neg2 = pp_pos.ffill()
pp_neg2 = sl.standardize(table=pp_neg2, start_labels_at_1=False)
print('Negatives standardized? ' + str(sl.is_standardized(pp_neg2)))

pp_all = pd.concat([pp_pos2, pp_neg2])

#print(pp_all.head())
#print(pp_all.tail())
```

    Positives standardized? True
    Negatives standardized? True
    


```python
pp_all.to_excel(main_folder + '\\' + 'all_pp_annots.xlsx')
```


```python
spec_cfg = load_audio_representation(spec_file, name="spectrogram")

dbi.create_database(output_file=db_name,  # empty brackets
                    dataset_name=r'test', selections=pp_all, data_dir=data_folder,
                    audio_repres=spec_cfg)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 142/142 [00:13<00:00, 10.81it/s]

    142 items saved to C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test\pierce_point_db.h5
    

    
    

## Step Three: Deploy Detector

### Copy Testing Files to Audio Folder


```python
annots = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test\all_pp_annots.xlsx')

annotsf = annots.ffill()

audio_folder = r'C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test\audio'

for idex, row in annotsf.iterrows():
    shutil.copyfile(annotsf.loc[idex]['filename'], audio_folder + '\\' + annotsf.loc[idex]['filename'].split('\\')[-1])

print('done')
```

    done
    

### Deploy Detector on Audio Data


```python
temp_folder = main_folder + '\\' + 'ringedS_tmp_folder'
threshold = 0.5
step_size = 1.0
batch_size = 16
buffer = 0.5 

base_folder = r'C:\Users\kzammit\Documents\Detector\detector-1sec'

model_names = [base_folder + '\\' + 'rs-1sec-1.kt', base_folder + '\\' + 'rs-1sec-2.kt', base_folder + '\\' + 'rs-1sec-3.kt', 
               base_folder + '\\' + 'rs-1sec-4.kt', base_folder + '\\' + 'rs-1sec-5.kt']

detections_csvs = [main_folder + '\\' + 'detections-raw-e-1.csv', main_folder + '\\' +  'detections-raw-e-2.csv', main_folder + '\\' + 'detections-raw-e-3.csv',
                  main_folder + '\\' + 'detections-raw-e-4.csv', main_folder + '\\' + 'detections-raw-e-5.csv']

audio_folder = r'C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test\audio'

for idx, model in enumerate(model_names):
    
    model = ResNetInterface.load(model_file=model, new_model_folder=temp_folder)
    
    audio_repr = load_audio_representation(path=spec_file)
    
    spec_config = audio_repr['spectrogram']
    
    audio_loader = AudioFrameLoader(path=audio_folder, duration=spec_config['duration'],
                                        step=step_size, stop=False, representation=spec_config['type'],
                                        representation_params=spec_config, pad=False)
    detections = pd.DataFrame()
    
    batch_generator = batch_load_audio_file_data(loader=audio_loader, batch_size=batch_size)
    
    for batch_data in batch_generator:
        # Run the model on the spectrogram data from the current batch
        batch_predictions = model.run_on_batch(batch_data['data'], return_raw_output=True)
    
        # Lets store our data in a dictionary
        raw_output = {'filename': batch_data['filename'], 'start': batch_data['start'], 'end': batch_data['end'],
                      'score': batch_predictions}
    
        batch_detections = filter_by_threshold(raw_output, threshold=threshold)
    
        # What do these labels represent? Is it 0 for no, and 1 for yes? why is 0 included in the
        detections = pd.concat([detections, batch_detections], ignore_index=True)
    
    detections.to_csv(detections_csvs[idx], index=False)
```

     48%|██████████████████████████████████████▎                                         | 143/299 [01:22<01:30,  1.72it/s]
      0%|                                                                                          | 0/299 [00:00<?, ?it/s]RuntimeWarning: Waveform padded with its own reflection to achieve required length to compute the stft. 73 samples were padded on the left and 0 samples were padded on the right
    100%|████████████████████████████████████████████████████████████████████████████████| 299/299 [00:21<00:00, 13.67it/s]
      0%|                                                                                          | 0/299 [00:00<?, ?it/s]RuntimeWarning: Waveform padded with its own reflection to achieve required length to compute the stft. 73 samples were padded on the left and 0 samples were padded on the right
    100%|████████████████████████████████████████████████████████████████████████████████| 299/299 [00:24<00:00, 12.31it/s]
      0%|                                                                                          | 0/299 [00:00<?, ?it/s]RuntimeWarning: Waveform padded with its own reflection to achieve required length to compute the stft. 73 samples were padded on the left and 0 samples were padded on the right
    100%|████████████████████████████████████████████████████████████████████████████████| 299/299 [00:24<00:00, 12.40it/s]
      0%|                                                                                          | 0/299 [00:00<?, ?it/s]RuntimeWarning: Waveform padded with its own reflection to achieve required length to compute the stft. 73 samples were padded on the left and 0 samples were padded on the right
    100%|████████████████████████████████████████████████████████████████████████████████| 299/299 [00:24<00:00, 12.34it/s]
      0%|                                                                                          | 0/299 [00:00<?, ?it/s]RuntimeWarning: Waveform padded with its own reflection to achieve required length to compute the stft. 73 samples were padded on the left and 0 samples were padded on the right
    100%|████████████████████████████████████████████████████████████████████████████████| 299/299 [00:24<00:00, 12.27it/s]
    

## Compare Results


```python
def compute_detections(labels, scores, threshold=0.5):
    """

    :param labels:
    :param scores:
    :param threshold:
    :return:
    """
    predictions = np.where(scores >= threshold, 1,0)

    TP = tf.math.count_nonzero(predictions * labels).numpy()
    TN = tf.math.count_nonzero((predictions - 1) * (labels - 1)).numpy()
    FP = tf.math.count_nonzero(predictions * (labels - 1)).numpy()
    FN = tf.math.count_nonzero((predictions - 1) * labels).numpy()

    return predictions, TP, TN, FP, FN
```


```python
output_dir = main_folder + '\\' + 'metrics'

db = dbi.open_file(db_name, 'r')

# Load the trained model
model = ResNetInterface.load(output_name, load_audio_repr=False, new_model_folder=temp_folder)

# Open the table in the database at the root level
table = dbi.open_table(db, '/test')

# Convert the data to the correct format for the model, and generate batches of data
gens = []

# not sure?
batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(table, "Table")))

# for the testing dataset
for group in db.walk_nodes(table, "Table"):
    generator = BatchGenerator(batch_size=batch_size, data_table=group,
                               output_transform_func=ResNetInterface.transform_batch, shuffle=False,
                               refresh_on_epoch_end=False, x_field='data', return_batch_ids=True)

    # attach the batches together so there's one for each dataset
    gens.append(generator)

gen = JointBatchGen(gens, n_batches='min', shuffle_batch=False, reset_generators=False, return_batch_ids=True)

scores = []
labels = []

for batch_id in range(gen.n_batches):
    hdf5_ids, batch_X, batch_Y = next(gen)

    batch_labels = np.argmax(batch_Y, axis=1)

    # will return the scores for just one class (with label 1)
    batch_scores = model.model.predict_on_batch(batch_X)[:, 1]

    scores.extend(batch_scores)
    labels.extend(batch_labels)

labels = np.array(labels)
scores = np.array(scores)

print('Length of labels is ' + str(len(labels)))

predicted, TP, TN, FP, FN = compute_detections(labels, scores, threshold)

print(f'\nSaving detections output to {output_dir}/')

df_group = pd.DataFrame()
for group in db.walk_nodes(table, "Table"):
    df = pd.DataFrame({'id': group[:]['id'], 'filename': group[:]['filename']})
    df_group = pd.concat([df_group, df], ignore_index=True)
df_group['label'] = labels[:]
df_group['predicted'] = predicted[:]
df_group['score'] = scores[:]
df_group.to_csv(os.path.join(os.getcwd(), output_dir, "classifications.csv"), mode='w', index=False)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
FPP = FP / (TN + FP)
confusion_matrix = [[TP, FN], [FP, TN]]
print(f'\nPrecision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('\nConfusionMatrix:')
print('\n[TP, FN]')
print('[FP, TN]')
print(f'{confusion_matrix[0]}')
print(f'{confusion_matrix[1]}')

print(f"\nSaving metrics to {output_dir}/")

# Saving precision recall and F1 Score for the defined thrshold
metrics = {'Precision': [precision], 'Recall': [recall], "F1 Score": [f1]}
metrics_df = pd.DataFrame(data=metrics)

metrics_df.to_csv(os.path.join(os.getcwd(), output_dir, "metrics.csv"), mode='w', index=False)

# Appending a confusion matrix to the file
row1 = ["Confusion Matrix", "Predicted"]
row2 = ["Actual", "RS", "Background Noise"]
row3 = ["RS", TP, FN]
row4 = ["Background Noise", FP, TN]
with open(os.path.join(os.getcwd(), output_dir, "metrics.csv"), 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(row1)
    writer.writerow(row2)
    writer.writerow(row3)
    writer.writerow(row4)

db.close()
```

    Length of labels is 426
    
    Saving detections output to C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test\metrics/
    
    Precision: 0.5
    Recall: 0.9154929577464789
    F1 Score: 0.6467661691542288
    
    ConfusionMatrix:
    
    [TP, FN]
    [FP, TN]
    [195, 18]
    [195, 18]
    
    Saving metrics to C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test\metrics/
    


```python
def confusion_matrix_plot(cf, output_folder,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=True):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        #plt.xlabel('Predicted label' + stats_text)
        plt.xlabel('Predicted label')
    else:
        plt.xlabel(stats_text)

    if title:
        #plt.title(title)
        #plt.title(stats_text)
        print('no title')

    plt.savefig(output_folder + '\\' + 'confusion_matrix.png')
```


```python
classifications_file = output_dir + '\\' + 'classifications.csv'

classifications = pd.read_csv(classifications_file)

cm = confusion_matrix_sklearn(classifications['predicted'], classifications['label'])

labels = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
categories = ['Ringed Seal', 'Noise']

#confusion_matrix_plot(cm, output_dir, group_names=labels, categories=categories,
#                 cmap=sns.diverging_palette(20, 220, as_cmap=True))

confusion_matrix_plot(cm, output_dir, group_names=labels, categories=categories,
                 cmap='viridis')
```

    no title
    


    
![png](output_17_1.png)
    



```python
scores = []
labels = []

for idx, det_file in enumerate(detections_csvs): 

    df = pd.read_csv(str(det_file))
    
    file_labels = df['label'].to_numpy()
    labels.append(file_labels)

    file_scores = df['score'].to_numpy()
    scores.append(file_scores)

avg_scores = np.transpose(scores).mean(axis=1)
avg_labels = np.transpose(labels).mean(axis=1)

labels_t = np.transpose(labels)
labels_df = pd.DataFrame(labels_t)

# Series of 1's and 2's, 2 means detector confused 
unq_labels = labels_df.nunique(axis=1)

# add column to dataframe 
labels_df['unql'] = unq_labels

# get indices 
indices = labels_df[labels_df['unql'] >= 2].index.tolist()
```


```python
# Set up an average df and output
avg_df = df
avg_df = avg_df.drop(columns='score')

# Create a column called "inc" for inconsistent
avg_df['inc'] = 'N'

# Create an average scores column 
avg_df['avg-score'] = avg_scores

# For each inconsistent index, set the score to N/A and inconsistent flag to Y
for idx, index in enumerate(indices): 
    avg_df['inc'][index] = 'Y'
    avg_df['avg-score'][index] = 'N/A'

avg_df.to_excel(main_folder + '\\' + 'ensemble-scores.xlsx', index=False)
```

    FutureWarning: Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas. Value 'N/A' has dtype incompatible with float64, please explicitly cast to a compatible dtype first.
    


```python
results_table = avg_df

#results_table = results_table[results_table.inc != 'Y']
#results_table = results_table[results_table.label == 1]

cols = ['filename']
results_table.loc[:,cols] = results_table.loc[:,cols].ffill()
results_table['Selection'] = results_table.index +1
results_table['View'] = 'Spectrogram 1'
results_table['Channel'] = 1
results_table['Begin Path'] = r'C:\Users\kzammit\Documents\Detector\detector-1sec\pp-test\audio' + '\\' + results_table.filename
results_table['File Offset (s)'] = results_table.start
results_table = results_table.rename(columns={"start": "Begin Time (s)", "end": "End Time (s)", "filename": "Begin File"})
results_table['Begin File'] = results_table['Begin File']
results_table['Low Freq (Hz)'] = 100
results_table['High Freq (Hz)'] = 1200

results_table.to_csv(main_folder + '\\' + 'raven_formatted_results.txt', index=False, sep='\t')
```


```python

```
