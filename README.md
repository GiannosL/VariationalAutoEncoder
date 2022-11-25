<h1>Variational AutoEncoder</h1>
<p>
    This is my first attempt at building a variational autoencoder (VAE). 
    I am using the MNIST dataset as a sample dataset to learn how to denoise images using a VAE.
</p>

<h2>Goals</h2>
<ul>
    <li>Learn how to build an AE</li>
    <li>Learn how to build a VAE</li>
    <li>Use hyper-parameter optimization</li>
    <li>Make my VAE work universally on datasets</li>
    <li>Make my VAE able to act both as a tool using <em>main.py</em> and as a package by importing it in another python script</li>
</ul>

<h2>AutoEncoder</h2>
<p>
    Autoencoders are a type of deep neural networks which can be used for dimensionality reduction. Autoencoders are composed of 
    two different neural networks, an encoder <em>e</em> and a decoder <em>d</em>. The autoencoder is trained to minimize 
    the difference between the input and the reconstruction using a kind of <em><strong>reconstruction loss</strong></em>. Because 
    the autoencoder is trained as a "whole" meaning that the encoder and decoder are trained at the same time.
</p>