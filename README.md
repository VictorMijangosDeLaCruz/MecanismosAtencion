# Introducción a los Mecanismos de Atención en Transformadores

## Objetivo
Los participantes en el taller conocerán los fundamentos teóricos esenciales para conocer los mecanismos de atención en redes neuronales, y serán capaces de implementar las unidades básicas de este tipos de mecanismos con el uso de Pytorch.

## Resumen
Las capas de atención son actualmente mecanismo centrales en los modelos del lenguaje. Los Transformers, que representan el estado del arte en este campo, se basan en el uso de capas de atención en combinación con otras estrategias. La atención también se ha utilizado en modelos basados en redes recurrentes de tipo sequence-to-sequence, brindando mejoras significativas en tareas de procesamiento de lenguaje natural como traducción automática y generación de texto. Entender cómo funcionan estos mecanismos es esencial para comprender los modelos del lenguaje actuales.

Este taller se propone presentar un primer acercamiento a los mecanismos de atención que se utilizan en las redes neuronales. En primer lugar, se presentarán los conceptos teóricos básicos para comprender la atención y su funcionamiento, se revisarán otros mecanismos de atención, principalmente la atención dispersa (sparse attention), y se verán la relación de la atención con los modelos del lenguaje auto-codificados y auto-regresivos. Finalmente, se planteará su relación con otros mecanismos como las capas convolucionales y las capas gráficas, resaltando sus ventajas y desventajas.

En segundo lugar, se abarcará los principios técnicos para la implementación de los mecanismos de atención en Pytorch y su incorporación dentro de la arquitectura de Transformers.

## Temario

1. <b>Introducción</b>
    1. [Motivación de la atención]
    2. [Atención en redes recurrentes](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/01RNNAtenttion.html)
2. <b>Atención en Transformadores</b>
    1. [Auto-atención](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/02SelfAttention.html)
    2. [Auto-atención y estructuras gráficas](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/03GraphAttention.html)
    3. [Embeddings y codificación posicional](https://victormijangosdelacruz.github.io/MecanismosAtencion/html/04Encoding.html)
