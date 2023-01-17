---
layout: post
title:  "Text simplification for the democratization of knowledge"
description: "Learning deep learning using transformers for text simplification"
date:   2021-07-17 14:17:19 +0100
categories: nlp
---


Language is surrounding us every minute of every day. One prerequisite of it is its understandability. So without understanding each other, language loses its purpose. It is not a coincidence that clean and clear speech and language are highly reputable. Only with that we can share knowledge and ideas over the borders of domains. Only with that can we make sure that information is spread democratically. In times where information will be the next essential resource - this is more important than ever. As the foundation of education, it must be equally available to everyone.

Technology and, in specific artificial intelligence can help us to reach this goal. Natural language processing is one of the most exciting machine learning fields, not just since the transformers. It allows us to automate complex language tasks that could be executed only by people with special knowledge. But as already said, knowledge should flow freely.

The project I will explain in the article should be one tiny step toward reaching this goal. It is designed to be the final project of the Data Science Retreat in which I participate. So this project is more about learning and discovering than about changing the world at once.
This article will be the first in a series of three. In this one, I will talk about the transformer implementation using the [HuggingFace](https://huggingface.co) library. The second one will be about the preparation of the dataset, etc. And the third one will be about a self-made-transformer. Together they are part of a bigger project. Check out the corresponding GitHub repo to get an overview.
Now let's get started!

There is no need to reinvent the wheel if there is HuggingFace 🤗. For everybody who doesn't know HuggingFace: it is one of the most famous and most used libraries for using transformers in NLP. I will not dive into the details, but I highly recommend checking out their stuff.

Using HuggingFace sequence-to-sequence models not only saved a lot of hassle - writing a transformer from scratch - but also opened a vast new horizon by using their pre-trained encoders and decoders.

### The Encoder-decoder model aka Sequence-to-sequence model
So everything starts with something like this:
<script src="https://gist.github.com/chrislemke/351cc2bdfaa951ddc501e239d096638d.js"></script>
We create an EncoderDecoderModel, initializing the encoder and the decoder with BERT (bert-base-uncased) freely available checkpoints. Which combination of encoder and decoder parameters is the best, varies from use-case to use-case. I started with using BERT's weights as encoder and decoder parameters and ended up using RoBERTa because of the more extensive corpus it was trained on. It is quite a task trying and analyzing all possible combinations. So we can be happy that Sascha Rothe, Shashi Narayan, and Aliaksei Severyn did this for us in their excellent paper.

Running the code above will give us some messages:
```
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel:
['cls.predictions.decoder.weight','cls.seq_relationship.weight', 'cls.predictions.transform.dense.bias',
'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight',
'cls.predictions.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or
with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be
exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertLMHeadModel:
['cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertLMHeadModel from the checkpoint of a model trained on another
task or with another architecture (e.g. initializing a BertForSequenceClassification model from
a BertForPreTraining model).
- This IS NOT expected if you are initializing BertLMHeadModel from the checkpoint of a model that you expect
 to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertLMHeadModel were not initialized from the model checkpoint at bert-base-uncased
and are newly initialized:
['bert.encoder.layer.7.crossattention.output.dense.bias', 'bert.encoder.layer.9.crossattention.self.query.bias'...]
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```
This message may seem confusing at the beginning. But actually, it just tells us that the CLS layers - which we don't need for our seq2seq model - are not initialized. It also tells us that a lot of weights from the cross-attention layers are initialized randomly. This makes sense if we look at the encoder (BERT), which does not have a cross-attention layer and therefore can't provide any parameters for it.
Now we already have an encoder-decoder model that provides the same functionality as other models of this type like BART, ProphetNet, or T5. The only difference is that our model is using pre-trained BERT weights.
The next step is setting up the tokenizer. HuggingFace also makes this step extremely simple. We need it now to share some parameters with the model configuration, but the main task we will see a bit later - once we talk about the training. With the help of the tokenizer, we can now configure some parameters of the model.

<script src="https://gist.github.com/chrislemke/1d4b229c1ccc6ba2c2641450c0316359.js"></script>

The exciting parameters to configure are the ones in the second block.
With `length_penalty` we push the model so that the simplified text is automatically shorter than the original text. The `num_beams` parameter is a bit more complicated to explain. In summary, it is about how many continuation words should be considered in the sequence to calculate the probability. Please check out this great block post to get a detailed picture of Beam search.

### The model
This is it! Our warm-started seq2seq model is now ready for fine-tuning. The next step is to set up all necessary training arguments. For a complete list, please refer to the documentation. I will only talk about some of them.

<script src="https://gist.github.com/chrislemke/b18d793d404cf141fea675c45e79a07c.js"></script>

`predict_with_generate` shall be one of them, with setting it to `true` metrics such as METEOR or ROUGE will be calculated while training. We will talk about metrics a bit later. It is important to know that loss and evaluation-loss as metrics are not as meaningful in text simplification as they are in other deep learning applications. 

For speeding up the training and decreasing the GPU's memory usage, we enable `fp16` to use 16-bit precision instead of 32-bit. We hope that this setting will not provoke gradient vanishing, which is dangerous when working with transformers. 

`gradient_accumulation_steps` goes in a similar direction. It determines how many updates steps are accumulated before the backward path is performed. When using big models on a single GPU, this is one possibility to not run instantly in a lack of GPU memory.
The `Seq2SeqTrainer`, which receives the training arguments as parameters, also expects a `compute_metrics` function.

<script src="https://gist.github.com/chrislemke/403fc1ff28258a14f9853cc8b46239a8.js"></script>

The METEOR metric (Metric for Evaluation of Translation with Explicit ORdering), as the name suggests, it was actually designed for translation. But also in text simplification, it is a practical value to evaluate the quality of the text. The metric is based on the harmonic mean of unigram precision and recall, with recall weighted higher than precision.
After extracting the `label_ids` and the `predictions` from the prediction object, we decode them. Line eleven makes sure that we correctly replace the pad token. After that, we calculate the METEOR and ROUGE metrics and return them as a dictionary.
Now everything is ready so that we can start the training. Depending on the dataset, this may take a while. My last approach with a 1.3M rows dataset took 60h. So let us wait and have a cookie 🍪.

### Evaluation
Finally, we can use our model to simplify some text.

<script src="https://gist.github.com/chrislemke/1efc5498f8eb9ffc78aec226003549cf.js"></script>

The input text should obviously be kind of complicated otherwise, simplification does not make sense. I picked the introduction to quantum mechanics from Wikipedia. That should be worth simplifying.

Let us go quickly over the code. Luckily it is again using HuggingFace:
In the eighth line, we load the model we just trained. We could have also just used the same instance from before. But we may not use it right away. For the same reason, we also load the tokenizer again. This time we can load it from our model.
Then we tokenize the input_text, so it is ready for the model to be processed. We give it a `max_length` of 60 - the rest will be padded or cut off - depending on the length.

With trained_model.generate, we generate the simplified text. Here we can play with the parameters like temperature or num_beams to improve the result.

### Next up
This concise introduction to one part of the project is just the beginning. I am already writing the following article about the dataset and all the typical work which was needed, so it fits the model. In the meantime, feel free to have a look at the GitHub repository. There you find all the code I was talking about in the article, as well as the code for the upcoming articles.
