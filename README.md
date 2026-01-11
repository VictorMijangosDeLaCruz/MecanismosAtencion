# Introducción a los Mecanismos de Atención en Transformadores

## Sobre el proyecto
El proyecto tiene como objetivo estudiar los sesgos inductivos, particularmente los sesgos inductivos relacionales en las capas de atención. El presente repositorio contiene material elaborado bajo el proyecto TA100924 que introduce las nociones básicas de las capas de atención, sus distintos tipos, su implementación con PyTorch, y la implementación de una arquitectura de transformers.

## Resumen
Las capas de atención son actualmente mecanismo centrales en los modelos del lenguaje. Los Transformers, que representan el estado del arte en este campo, se basan en el uso de capas de atención en combinación con otras estrategias. La atención también se ha utilizado en modelos basados en redes recurrentes de tipo sequence-to-sequence, brindando mejoras significativas en tareas de procesamiento de lenguaje natural como traducción automática y generación de texto. Entender cómo funcionan estos mecanismos es esencial para comprender los modelos del lenguaje actuales.

Este repositorio se propone presentar un primer acercamiento a los mecanismos de atención que se utilizan en las redes neuronales. En primer lugar, se presentarán los conceptos teóricos básicos para comprender la atención y su funcionamiento, se revisarán otros mecanismos de atención, principalmente la atención dispersa (sparse attention), y se verán la relación de la atención con los modelos del lenguaje auto-codificados y auto-regresivos. Finalmente, se planteará su relación con otros mecanismos como las capas convolucionales y las capas gráficas, resaltando sus ventajas y desventajas.

En segundo lugar, se abarcará los principios técnicos para la implementación de los mecanismos de atención en Pytorch y su incorporación dentro de la arquitectura de Transformers.

## Temario

1. <b>Introducción</b>
    1. [Atención en redes recurrentes](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/01RNNAtenttion.html)
2. <b>Auto-atención en Transformadores</b>
    1. [Auto-atención](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/02SelfAttention.html)
    2. [Auto-atención y estructuras gráficas](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/03GraphAttention.html)
3. <b>Otras capas dentro de los Transformadores</b>
    1. [Embeddings y codificación posicional](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/04Encoding.html)
    2. [Suma y normalización](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/05Normalization.html)
4. <b>Multi-cabeza y atención enmascarada</b>
    1. [Cabezas de atención](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/06AttentionHead.html)
    2. [Auto-atención enmascarada](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/07MaskedAttention.html)
    3. [Atención dispersa](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/07bSparseAtt.html)
9. <b>Construcción del transformador</b>
    1. [Sobre el gradiente en capas de atención](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/08aGradientAtt.html)
    2. [Optimizador Noam](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/08Noam.html)
    3. [Transformador](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/09FullTranformer.html)
10. <b> Arquitecturas específicas</b>
    1. [Arquitectura encoder-only (BERT)](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/BERT.html)
    2. [Ajuste fino de BETO (BERT es español)](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/BETO_Example.html)

### Notebooks

Los notebooks utilizados pueden encontrarse [aquí](https://victormijangosdelacruz.github.io/MecanismosAtencion/Notebooks/).

La presentación de la escuela de verano 2024 puede encontrarse [aquí](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/2024School_Transformers-1.pdf).

## Referencias

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
- Bjorck, N., Gomes, C. P., Selman, B., & Weinberger, K. Q. (2018). Understanding batch normalization. Advances in neural information processing systems, 31.
- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478.
- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Fiedel, N. (2023). Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240), 1-113.
- Cordonnier, J. B., Loukas, A., & Jaggi, M. (2020). Multi-head attention: Collaborate instead of concatenate. arXiv preprint arXiv:2006.16362.
- Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does bert look at? an analysis of bert's attention. arXiv preprint arXiv:1906.04341.
- Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509.
- Dar, G., Geva, M., Gupta, A., & Berant, J. (2022). Analyzing transformers in embedding space. arXiv preprint arXiv:2209.02535.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.
- Ioffe, S., & Szegedy, C. (2015, June). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning (pp. 448-456).
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
- Knyazev, B., Taylor, G. W., & Amer, M. (2019). Understanding attention and generalization in graph neural networks. Advances in neural information processing systems, 32.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568, 127063.
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. arXiv preprint arXiv:1803.02155.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. arXiv preprint arXiv:1710.10903.

### Otros recursos

- [The Annotated Transformer (tutorial)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Transformers are graph neural networks (blog)](https://thegradient.pub/transformers-are-graph-neural-networks/)
- [The math behind transformers (video)](https://www.youtube.com/watch?v=UPtG_38Oq8o&t=1s)
- [Las matemáticas de los transformers (video)](https://www.youtube.com/watch?v=w5pfPvGGSIY&t=8s)

------------------------------------------------------------------------------------------------------

<p style="text-align: right;">Material desarrollado con apoyo del proyecto PAPIIT TA100924</p>
