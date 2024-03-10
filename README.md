# Metric learning in analyzing political bias
Analyzing the political bias and reliability of articles using metric learning techniques and DeBERTa

In bias estimation using metric learning, we apply the
discriminative approach and fine-tune a DeBERTa transformer
model to map both texts and their descriptions (’Left’, ’Right’
etc.) to the same joint space. This is inspired by multi-modal
joint embedding approach, where common embeddings are
learned for input from different domain, i.e. images and their
text descriptions.
