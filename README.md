
# Multilabel metric learning in analyzing political bias
Analyzing the political bias and reliability of articles using metric learning techniques and DeBERTa.
[Notebook with code](https://github.com/luftpro/political-bias-metric-deberta/blob/main/deberta-metric.ipynb)

## Dataset
[CLEF CheckThat! Lab 2023 Task 3 dataset](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab/-/tree/main/task3y)

## Methodology
In bias estimation using metric learning, we apply the
discriminative approach and fine-tune a DeBERTa transformer
model to map both texts and their descriptions (’Left’, ’Right’
etc.) **to the same joint space**. This is inspired by multi-modal
joint embedding approach, where common embeddings are
learned for input from different domain, i.e. images and their
text descriptions.

Six descriptions were used in training the multilabel classifier: Left, Right, Liberal, Conservative, Credible, Unreliable

## Easy positive&mdash;semi-hard negative triplet mining
Easy positive is a sample that corresponds to the anchor and is the closest, whereas a semi-hard negative is a sample that lies further than the positive sample but within the defined margin from it.

## Triplet loss calculation
We use a Triplet Loss as defined in visual-semantic joint embedding learning with easy positive&mdash;semi-hard negative triplet mining which demonstrated great results compared to usual batch all or hard negative approaches. For a given text embedding $\mathcal{T}$, let $`\mathcal{D}_{EP}`$ be the easy positive description embedding and $`\mathcal{D}_{SH}`$ be the semi-hard negative description embedding. Using Euclidean distance $d(a,b)$, a triplet loss for a text embedding as the anchor is:
```math
L_{\mathcal{T}} = [d(\mathcal{T},\mathcal{D}_{EP})-d(\mathcal{T},\mathcal{D}_{SH})+\alpha]_+
```

Since both text and description embeddings are mapped to the same space, we also calculate the loss for when a description is used as the anchor:
```math
L_{\mathcal{D}} = [d(\mathcal{D},\mathcal{T}_{EP})-d(\mathcal{D},\mathcal{T}_{SH})+\alpha]_+
```

In the end, the total loss for a minibatch is defined as follows:
```math
L_{total} = \frac{1}{N}\sum_{i}^{N}L_{\mathcal{T}_i} + \frac{1}{M}\sum_{j}^{M}L_{\mathcal{D}_j},
```
where $N$ is the batch size and $M$ is the number of overall possible descriptions (labels).

A prediction for a website is done by taking the mean of the distances from the texts of this website to their predicted descriptions and choosing _k_ closest descriptions. Embeddings for the texts and descriptions are normalized together.


## Results
The model achieved an average f1-score of **0.37** across six different labels, performing better than DeBERTa trained classically.
