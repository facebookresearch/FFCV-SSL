(February 8, 2023)

### Addition to the FFCV library

* Multi views: FFCV-SSL is able to return an arbitraty number of different view of a given field with different pipelines.
* More augmentations: Data augmentations is crucial for methods like Self-supervised learning, in this fork we add ColorJitter, Solarization, Grayscale, Rotation.. 
* Seeding: Being able to fix the seed of a given transformation is important for reproducibility
* Data augmentations parameters: In this fork you can get label pipelines that will return the parameters of the data augmentations that are used.
* Add a SequentialContiguous ordering to split a list of indices into a continuous list of indices for each gpu (Instead of putting each data point on different gpu which is the default behaviour of Sequential).
* Add a padding function to pad images with differents resolution to a fixed size.


