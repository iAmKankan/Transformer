
### 3.5 Positional Encoding
Since our model contains no [recurrence][1] and no [convolution][2], for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention. Layer Type Complexity per Layer Sequential Maximum Path Length Operations Self-Attention O(n 2 · d) O(1) O(1) Recurrent O(n · d 2 ) O(n) O(n) Convolutional O(k · n · d 2 ) O(1) O(logk(n)) Self-Attention (restricted) O(r · n · d) O(1) O(n/r) 


The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [8]. 

In this work, we use sine and cosine functions of different frequencies: 

$$\Large {\color{Purple}\begin{matrix*}
 \textrm{PE}(pos, 2i) &=& sin \Big( \dfrac{pos}{10000 ^ {2i / d}} \Big)  \\
 \textrm{PE}(pos, 2i + 1) &=& cos\Big( \dfrac{pos}{10000 ^ {2i / d}} \Big) 
\end{matrix*}} $$


where $pos$ is the position, and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a **sinusoid**. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesised it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos. We also experimented with using learned positional embeddings [8] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.


[1]: https://share.google/aimode/qveb3RiTDnFjn4gCf "Recurrence in a neural network is a mechanism where the output from a previous step is fed back into the network as input for the current step. This creates a loop, allowing the network to maintain a memory of past events to process sequential or time-series data"

[2]: https://share.google/aimode/jAeQkUNQGnYnHz07x "In a neural network, convolution is a mathematical operation that applies a sliding filter (kernel) over an input grid—such as an image—to extract specific features like edges, colors, or patterns" 
