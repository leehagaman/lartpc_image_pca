
The idea is to train a decoder on LArTPC images and look at what the most important features (principal components) are. Neutrino energy? numuCC vs nueCC? muon track angle? Cosmic ray density?

Using MicroBooNE public data from https://zenodo.org/records/8370883, inclusive, without wire information. Images are real cosmic rays overlaid with simulated neutrino interactions. I crop around the true neutrino location, and only use the collection plane.

Inspired by this code, doing the same thing with human faces: https://github.com/HackerPoet/FaceEditor
