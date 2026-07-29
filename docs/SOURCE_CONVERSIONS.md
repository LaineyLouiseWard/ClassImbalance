# Source conversions — index

**The conversions themselves live in `references_md/`, which is gitignored** (it holds copyrighted
full texts, same reason `papers-md` is a private repo). This index is tracked so the mapping survives;
the files it points at are local only.

Generated 2026-07-29. One row per **cited** entry in `manuscript_v2/Bibliography.bib`.
Conversions live in this directory so they travel with the repo. The global library is
`~/Documents/Github/papers-md/` (MinerU pipeline: `papers-md/pipeline/convert.py`).

**Rule: do not cite a source that has no conversion here and has not been read.**

25 cited · 63 conversions present · 38 not currently cited

| Citekey | Title | Conversion |
|---|---|---|
| `chengBoundaryIoUImproving2021` | Boundary IoU: Improving Object-Centric Image Segmentation Evaluation | `cheng-2021-boundary-iou-improving-object-centric-segmentation-evaluation.md` |
| `csurkaWhatGoodEvaluation2013` | What is a good evaluation measure for semantic segmentation? | `csurka-2013-good-evaluation-measure-semantic-segmentation.md` |
| `dengImageNetLargescaleHierarchical2009` | ImageNet: A Large-Scale Hierarchical Image Database | `deng-2009-imagenet-large-scale-hierarchical.md` |
| `kangDecouplingRepresentationClassifier2020` | Decoupling Representation and Classifier for Long-Tailed Recognition | `kang-2019-decoupling-representation-and-classifier-for-longtailed-recognition.md` |
| `kattenbornSpatiallyAutocorrelated2022` | Spatially autocorrelated training and validation samples inflate perfo | `kattenborn-2022-spatially-autocorrelated-training-validation-cnn.md` |
| `kohliRobustHigherOrder2009` | Robust Higher Order Potentials for Enforcing Label Consistency | `kohli-2009-robust-higher-order-potentials-label-consistency.md` |
| `krawczykLearningImbalancedData2016` | Learning from imbalanced data: open challenges and future directions | `krawczyk-2016-learning-from-imbalanced-data.md` |
| `lambertMSegCompositeDataset2020` | MSeg: A Composite Dataset for Multi-Domain Semantic Segmentation | `lambert-2020-mseg-composite-dataset-for-multidomain-semantic-segmentation.md` |
| `linFocalLossDense2017` | Focal Loss for Dense Object Detection | `lin-2017-focal-loss-for-dense-object-detection.md` |
| `liuSwinTransformerHierarchical2021` | Swin Transformer: Hierarchical Vision Transformer using Shifted Window | `liu-2021-swin-transformer-hierarchical.md` |
| `loshchilovSGDRStochasticGradient2017` | SGDR: Stochastic Gradient Descent with Warm Restarts | `loshchilov-2017-sgdr-warm-restarts.md` |
| `maxwellThematicClassificationAccuracy2020` | Thematic Classification Accuracy Assessment with Inherently Uncertain  | `maxwell-2020-thematic-classification-accuracy-assessment-uncertain-boundaries.md` |
| `milletariVNetFullyConvolutional2016` | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Imag | `milletari-2016-vnet-fully-convolutional-neural-networks-for-volumetric-medical.md` |
| `montgomeryDesignAnalysisExperiments2017` | Design and Analysis of Experiments | `montgomery-2012-ch6-2k-factorial-design.md` |
| `reinaSystematicEvaluationImage2020` | Systematic Evaluation of Image Tiling Adverse Effects on Deep Learning | `reina-2020-systematic-evaluation-image-tiling-adverse-effects-deep-learning-2.md` |
| `robertsCrossValidationStrategies2017` | Cross-validation strategies for data with temporal, spatial, hierarchi | `roberts-2017-cross-validation-strategies-temporal-spatial-hierarchical.md` |
| `saadeldinUsingDeepLearning2022` | Using deep learning to classify grassland management intensity in grou | `saadeldin-2022-using-deep-learning-classify-grassland-management-intensity-2.md` |
| `volpiDenseSemanticLabeling2017` | Dense Semantic Labeling of Subdecimeter Resolution Images With Convolu | `volpi-2017-dense-semantic-labeling-subdecimeter-resolution-cnn.md` |
| `wangUNetFormerUNetlikeTransformer2022` | UNetFormer: A UNet-like transformer for efficient semantic segmentatio | `wang-2022-unetformer-unetlike-transformer-for-efficient-semantic-segmentation-6.md` |
| `wardLabelQualityCeilingCode2026` | Code for ``Diagnosing a Label-Quality Ceiling in Imbalanced Rural Land | **MISSING** |
| `xiaOpenEarthMapBenchmarkDataset2023` | OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cove | `xia-2023-openearthmap-benchmark-dataset-for-global-highresolution-land-cover-8.md` |
| `xiaoUnifiedPerceptualParsing2018` | Unified Perceptual Parsing for Scene Understanding | `xiao-2018-unified-perceptual-parsing.md` |
| `yuanEvaluationPretrainingImpact2019` | High-Resolution Remote Sensing Imagery Classification of Imbalanced Da | `yuan-2019-evaluation-pretraining-impact-finetuning-for-remote-sensing-scene-2.md` |
| `zhangLookaheadOptimizerSteps2019` | Lookahead Optimizer: k steps forward, 1 step back | `zhang-2019-lookahead-optimizer-steps-forward-step-back.md` |
| `zhouSemanticUnderstandingScenes2019` | Semantic Understanding of Scenes Through the ADE20K Dataset | `zhou-2019-semantic-understanding-ade20k.md` |

## Cited with no conversion

None. Every cited paper has a conversion in this directory.

`wardLabelQualityCeilingCode2026` is the only cited entry without one, and correctly so — see the note
below.

## Not yet obtainable, so not yet citable

These were proposed as additions but are **absent from Zotero, from `papers-md/` and from this
directory**, so nobody has read them. Under the convert-before-citing rule they cannot go in the
manuscript until obtained → added to Zotero → converted with MinerU → read.

| Proposed for | Work | Status |
|---|---|---|
| AdamW optimiser | Loshchilov & Hutter 2019, *Decoupled Weight Decay Regularization*, ICLR 2019 (no DOI; the arXiv DOI `10.48550/arXiv.1711.05101` belongs to the 2017 preprint and must not be attached) | Freely available — obtainable |
| Mantel correlogram method | Legendre & Legendre, *Numerical Ecology*, 3rd English ed. | **A ~1000-page book.** Not a paper; cannot be converted or read this way. The "sec. 13.1.6" in `scripts/analysis/spatial_correlogram.py` is unverified |
| Hellinger transform | Legendre & Gallagher 2001, *Oecologia* | Not held; obtainable only if the PDF can be retrieved |

**Mantel 1967 is not the right substitute** — it gives the bare two-matrix permutation test, whereas
the script computes a correlogram (distance classes, per-class binary model matrices, sign reversal,
half-distance cutoff, progressive Holm). Citing it would misattribute the method.

**The low-friction alternative for the correlogram:** `kattenbornSpatiallyAutocorrelated2022` is
already cited, already converted and already read, and is the paper this analysis explicitly mirrors
(`spatial_correlogram.py` docstring). It can carry the method reference without any new sourcing.

## Note

`wardLabelQualityCeilingCode2026` is the authors' own Zenodo software deposit — there is no
paper to convert, and no conversion is expected.
