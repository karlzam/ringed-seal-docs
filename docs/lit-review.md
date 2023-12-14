# Lit Review

## Lakshminarayanan et al. 2017 - Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

Summary: **Train an ensemble, some other stuff about adversarial training and proper scoring, fig 2 useful**

Notes:

- NNs tend to produce overconfident predictions 
- calibration: frequentist notion of uncertainty which measures the discrepancy between subjective forecasts and
(empirical) long-run frequencies. The quality of calibration can be measured by proper scoring rules
[17] such as log predictive probabilities and the Brier score [9].
- generalization of the predictive uncertainty to domain shift (also referred to as out-of-distribution examples [23]), that is,
measuring if the network knows what it knows.
- Bayesian model averaging (BMA) assumes that the true model lies within
the hypothesis class of the prior, and performs soft model selection to find the single best model within
the hypothesis class [43]. In contrast, ensembles perform model combination, i.e. they combine the
models to obtain a more powerful model; ensembles can be expected to be better when the true model
does not lie within the hypothesis class
-  (1) use a proper scoring rule as the training criterion, (2) use adversarial
training [18] to smooth the predictive distributions, and (3) train an ensemble


## Padovese et al. 2023 - Adapting deep learning models to new acoustic environments - A case study on the North Atlantic right whale upcall

Summary: **NARW, Transfer Learning, ResNet vs VGG, 3 different locations**

Notes: 

- ResNet outperformed VGG
- talks about the limitations very well
- need large amounts of data 
- model generalization is difficult 
- difficult when similar vocalizations are produced by different species 
- task remains the same while the environment changes 
- open source, python command line interface
- current state of the art detectors use mag spectrograms 
- t-SNE on last layer to determine if the network has really learned the features
- Lower FPR is better 

## White et al. 2023 - One size fits all? Adaptation of trained CNNs to new marine acoustic environments

Summary: **Fine tuning of CNN on distinctly different soundscape, odontocete, pretrained efficient net**

Notes: 

- Fine tune "small scale CNN"
- Fine tuning significantly improves  the performance of the CNN
- Each channel of the input RGB image corresponds to a single spectrogram computed at one of three different time-frequency resolutions (frequency bins of widths 93.75 Hz, 46.88 Hz and 23.44 Hz corresponding to FFT sizes of 1024, 2048 and 4096) standardized for the sampling rate 96 kHz. The spectrogram values are standardized to correspond to the range − 80 to 0 dBfs.
- Keep feature extractor frozen, "dropconnect" employed
- GradCam for saliency maps
- local fine-tuning is essential to capture the variability in local soundscape characteristics

## Kirsebom et al. 2020 - Performance of a deep neural network at detecting North Atlantic right whale upcalls ##

Summary: **ResNet, Binary, NARW, compared to LDA, looked at SNR**

Structure of paper: 

Intro:

  - Introduce species 
  - Explain area 
  - Explain motivation for conservation efforts (vessel speed, etc)
  - Explain why PAM is needed 
  - Explain classification algorithms and older ones 
  - Explain performance of these classification algorithms 
  - Introduce ML
  - Introduce DL and example applications
  - CNNs with examples and spectrogram ones
- Acoustic Data Collection
  - Dates, locations, map
  - Equipment type and duty cycle 
  - Differences in data recorded, SNR variability
- Generation of training datasets, neural network design, training protocol
  - First looked at detector for NARW previously in existence 
  - Talked about annotations and where they got them from
  - Extracted segment length, centered on call
  - Train test split, used a t val to split to an 85:15 split, cool diagram
- Spectrogram and SNR Computation
  - Downsample rate
  - scale (dB)
  - window size, step size, window function
  - No effort to correct for systematic difference in signal amp
  - Estimate SNR of each sample (with steps)
- NN Architecture: 
  - binary classification (define the classes)
  - duration of spectrograms 
  - brief mention of the CNN architecture based on He et al 2016
  - how passed to fully connected network and gives probabilities 
- Training Protocol:
  - GPU with memory stats 
  - Batch sizes with termination after a preset number of epochs 
  - optimizer, with recommended vals (lr 0.001, decay 0.01, b1 of 0.9, b2 of 0.999)
  - Didn't tune hyperparameters 
  - network trained to maximize the F1 score
  - trained using 5-fold cross validation, N=100
  - Then trained on full data without cross validation 
  - Repeated 9 times with different random gen seeds to assess the sensititivty of the training outcome to initial conditions
- Compared to linear discriminant analysis
- Detection algorithm:
  - detect in continuos data we:
    - segment to window size of 3s with step size of 0.5s
    - each segment passed into DNN classifier 
    - found best to smoothen the classification scores using a 2.5s wide averaging window 
    - greatly reduced the number of false positives
    - applied detection threshold
  - to compute recall, precision, false-positive:
    - merged adjacent bins into detection events 
    - had a temporal buffer
    - false positive when annotated overlap was less than 50%
- Results of detection and classification tasks
  - 9 independent training sessions 
  - looked at increasing size of dataset 
  - looked at discarding low SNR samples
  - plots of precision and recall vs LDA
- Summary & conclusion
  - Discussion:
    - DNN have larger capacity to handle variance in the data than LDA
    - General review on what was already talked about
    - Results summarized
    - A brief discussion on papers that were released during the writing of their paper
  - Conclusion: 
    - Shown that DNNs work for the task
    - improved performance by expanding variance of dataset
    - "consider a wider temporal context"

Notes:

- Classical methods plateau at 50% recall when false detections are kept below 10%
- Last decade, ML are now the way to go
- CNNs have been used to analyze info in spectrograms (with examples)
- Locations vary in flow noise, background, strum, knocks, tidal currents, surface motion
- SNR variability due to different locations 
- Statement about not enhancing SNR before feeding to neural network
- Quasi-random time shifts are desirable for DNN classifier bc they encourage network to learn a more general, time translation invariant, representation of the upcall 
- F1 score defined as the "harmonic mean of precision and recall", attaches equal importance to the two

