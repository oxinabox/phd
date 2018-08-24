Title: On the surprising capacity of linear combinations of embeddings for natural language processing

Word Embeddings have, in a big way, changed how we do natural language processing.
We use them (implicitly or explicitly) as a component in a variety of powerful systems:
from LSTMs to complicated tree structured models.
These models are very exciting, and work well to push the envelope of what can be done.
But there are also simpler models: systems based on just summing the word embeddings.
On a variety of tasks, these work very well -- often better than the more complex models.

This thesis examines linear combinations of embeddings for natural language understanding tasks.
In particular simple sums, but also some weightings such as means.
The thesis contains published works demonstrating the utility of the linear combinations of embeddings
for representing sentences, short phrases, word senses and usage contexts.
It also contains works investigating the extent that the original input words and sentences can be recovered from the summed embedding representation.

In brief, it is found that a sum of embeddings is a particularly effective dimensionality-reduced representation of a bag of words.
The dimensionality reduction is carried out at the word level via the implicit matrix factorization 
on the collocation probability matrix.
It thus captures into the dense word embeddings the key features of lexical semantics:
words that occur in similar contexts have similar meanings.
We find that summing these representations of words gives us a very useful representation of structures built upon words.

A limitation of the sum of embedding representation is that it is unable to represent word order.
This representation does not capture any order related information; unlike for example a recurrent neural network.
Recurrent neural networks, and other more complex models, are out performed by sums of embeddings in tasks where word order is not highly significant.
It is found that even in tasks were word order does matter to an extent, the improved training capacity of the simpler model still can mean that it performs better than more complex models.
This limitation thus hurts surprisingly little.


