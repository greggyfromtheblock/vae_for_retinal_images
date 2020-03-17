# Overview
## Problem statement
- learn representations from retina fundus image datasets using deep neural networks
- evaluate learnt latent space regarding cvd (cardio vascular disease) risk factors
## Proposed work
- make use of unsupervised learning methods such as autoencoders to extract relevant features
- make use of visualization techniques to analyze latent space
# Approach
- Why this approach?
- make use of convolutions for the extraction process as they are state of the art for visual feature extraction from image data
- use of variational autoencoder technique as they squeeze relevant features in a reduced dimension space
- use of algorithms such as PCA, UMAP to visualize the learnt representations
- Which other approaches considered?
- GANs, standard autoencoders
- Why other approaches not taken?
- focus of the project is to use learnt representation. in terms of image reconstruction techniques like generative adversarial networks could also be considerable but that is not the focus
- standard autoencoders tend to overfit, VAEs learn distributions to avoid overfitting
# project methodology
- iterative and incremental approach for work on work packages
# Work packages
## WP1: workflow
- workflow setup
- goal: bundling scripts for structure and automated execution
- proposed workflow tool: snakemake
- scripting work: workflow
## WP2: preprocessing
- analysis of raw datasets
- investigate preprocessing and augmentation steps
- investigate when to do each of the preprocessing and augmentation steps (pre vs on-the-fly)
- scripting work: preprocessing
## WP3: modelling
- use of torchsupport
- iterative process of development: code sceletons for each class, stepwise implementation
- scripting work: dataset and dataloading, model and layers, training procedure
## WP4: evaluation
- investigate analysis process (what and how do we analyze our trained model / the latent space)
- investigate visualization techniques
- iterative process of development: code sceletons for each class, stepwise implementation
- scripting work: clustering, visualization (PCA, tSNE, UMAP, ...)
# Dependencies & risks
- COVID-19!!!!
## Stakeholders
Who are the critical stakeholders? Regarding what topics?
-
## Risks
What are the risks?
-
## Dependencies
What are the dependencies? Include both internal (other teams) and external (partners).
-
# Success criteria
The criteria that must be met to consider this project a success.
- weekly status meetings, code reviews
- documentation of work
- clean github repo
# initial time line (note: these were our goals by end of each week. we work iteratively which means it is totally ok and necessary to work on multiple work packages each week)
- week 1: project setup, WP1
- week 2: WP1 (preprocessing), WP2 (encoding part: cnns to latent, if needed also for supervised tasks)
- week 3: buffer
- week 4: WP2 (decoding part: latent to reconstruction and full VAE model)
- week 5: plotting, latent analysis
- week 6: clustering
