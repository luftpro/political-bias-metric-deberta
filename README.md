<script type="text/javascript" async
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Metric learning in analyzing political bias
Analyzing the political bias and reliability of articles using metric learning techniques and DeBERTa

# Methodology
In bias estimation using metric learning, we apply the
discriminative approach and fine-tune a DeBERTa transformer
model to map both texts and their descriptions (’Left’, ’Right’
etc.) to the same joint space. This is inspired by multi-modal
joint embedding approach, where common embeddings are
learned for input from different domain, i.e. images and their
text descriptions.

# Mining of triplets
We use a Triplet Loss as defined in visual-semantic joint embedding learning~\cite{contrastembed} with easy positive---semi-hard negative triplet mining which demonstrated great results compared to usual batch all or hard negative approaches~\cite{eptripletmining}. Easy positive is a sample that corresponds to the anchor and is the closest, whereas a semi-hard negative is a sample that lies further than the positive sample but within the defined margin from it. For a given text embedding $\mathcal{T}$, let $\mathcal{D}_{EP}$ be the easy positive description embedding and $\mathcal{D}_{SH}$ be the semi-hard negative description embedding. Using Euclidean distance $d(a,b)$, a triplet loss for a text embedding as the anchor is:
\begin{equation}
L_{\mathcal{T}} = [d(\mathcal{T},\mathcal{D}_{EP})-d(\mathcal{T},\mathcal{D}_{SH})+\alpha]_+
\end{equation}

Since both text and description embeddings are mapped to the same space, we also calculate the loss for when a description is used as the anchor:
\begin{equation}
L_{\mathcal{D}} = [d(\mathcal{D},\mathcal{T}_{EP})-d(\mathcal{D},\mathcal{T}_{SH})+\alpha]_+
\end{equation}

In the end, the total loss for a minibatch is defined as follows:
\begin{equation}
L_{total} = \frac{1}{N}\sum_{i}^{N}L_{\mathcal{T}_i} + \frac{1}{M}\sum_{j}^{M}L_{\mathcal{D}_j},
\end{equation}
where $N$ is the batch size and $M$ is the number of overall possible descriptions (labels).

A prediction for a website is done by taking the mean of the distances from the texts of this website to their predicted descriptions and choosing \textit{k} closest descriptions. Embeddings for the texts and descriptions are normalized together.


# Results
The model achieved an average f1-score of 0.37 across six different labels.
