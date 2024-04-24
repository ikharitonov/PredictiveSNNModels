The goal of this module is to collate Heidelberg digits into pairs/sequences (e.g. [1,2], [3,4] or [0,1,2,3,4,5,6,7,8,9], [7,8,9,0,1,2,3,4,5,6]). Here, the strategy is the same as in __mnist_digit_snn__ and <a href=https://www.cell.com/patterns/fulltext/S2666-3899(22)00271-9>here</a> - to use L1 loss on membrane potential and hopefully achieve predictive functionality in the network.

__digits_snn.ipynb__ contains intial work on the spiking inputs, particularly considerations of representing them in different formats (e.g. original, plt.eventplot-friedly, tensor-friendly), resampling, justifications for parameter choices (for digit standardisation purposes).

__snn_project1.ipynb__ is a more in-depth exploration of different approaches in regards to inputs (mainly, the simpler pair of digits approach), visualisations, trained model assessment.

__model.py__ is mainly inspired by <a href=https://github.com/fzenke/spytorch/tree/main>Friedemann Zenke's SpyTorch</a> - a recurrent network of leaky integrate-and-fire (LIF) units with synaptic decay is implemented.

__digits_io.py__ implements all input handling functions.
