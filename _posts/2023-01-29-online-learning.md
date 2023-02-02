---
layout: post
title:  "Online learning"
description: "The unpopular but cool brother of batch learning"
date:   2023-01-29 17:43:19 +0100
categories:
- machine learning
- online learning
---

Let's talk about learning for a moment. How does a machine learn? How does a human learn? What do they share and where do they differ?

What is called batch learning in the context of ML is nothing more than learning from provided sources such as CSV files, databases, etc. In the human context, this is like learning vocabulary for a foreign language. But people also learn in many other ways. One very successful one is learning by doing. This applies not only to mundane activities like playing soccer or writing software but also to something as complex as human interaction. The more often and longer you pursue such an activity, the better you become. This happens almost automatically and everyone knows the feeling that a certain action, which once seemed difficult, is now quite easy.

But what about machine learning? In order to continuously improve a model that was created with batch training, regular training is required. But this training starts from scratch every time. The model does not build on the already collected "knowledge". Generally, this is not a problem. An automated regular training of a model is nothing special and with those like SageMaker, etc. the implementation is also quite simple. But it is still batch learning. But in the ML world, there is another type of learning: online learning.

## Keep learning 🙇‍♂️
What is "online learning" and why haven't I heard about it? I recently asked myself the same questions. For all those who are looking for quick answers:

> In computer science, online machine learning is a method of machine learning in which data becomes available in a sequential order and is used to update the best predictor for future data at each step, as opposed to batch learning techniques which generate the best predictor by learning on the entire training data set at once.

Source: [Wikipedia](https://en.wikipedia.org/wiki/Online_machine_learning)

You probably haven't heard of it, because in industry and academia, it plays only a small role compared to batch learning. The [scikit-learn](https://scikit-learn.org/stable/index.html) library - as a representative of batch learning is so elaborate, well-known and omnipresent. Besides this, well-known online-learning libraries like [river](https://riverml.xyz/0.15.0/) are only small fish.

Let's go more into detail to really understand what online learning is and how it works. This form of supervised learning fulfills what many machine learning borrowers fundamentally expect from ML/AI. Continuous learning - just as we humans do ourselves. Let's take a closer look at a practical example: anomaly detection for fraud prevention. There are many different algorithms that can be used here, but for our example, we want to work with a simple binary classifier - so strictly speaking, this is not anomaly detection at all. But you know what I mean.

## Example 🦄
The setting for this example is a booking process where customers can book flying unicorns. And as with all online services, there are always a few black sheep. So <span style="color:pink">Unifly</span> - that's the name of the company - has the plan to implement a fraud prevention system. For reasons we will discuss later, they have decided to operate an online learning classifier. The way it works is simple: a wide variety of data is collected throughout the booking process, such as the customer's email, the browser used, or how long the customer stays on certain screens. Shortly before the payment process, this data is merged and the model determines whether it is a potential fraud attempt.

I get that. But how is the model trained if not with predetermined training data? - Once the booking is processed after the model has done its prediction the booking is either marked as fraud or no fraud - probably done by humans 🧬. And this is the data that is used for online learning. The model just needs to receive the information if the booking was fraudulent or not. That's it! No need for training the model with collected data. No need to re-train the model on new data. The model is continuously learning and improving itself.

## Challenges 🤔
Of cause, there are some drawbacks to this approach. Whenever a model is continuously learning it is/was quite "stupid" at the beginning. So before it is put in production it may be either learned with collected data or it runs silently without predicting anything but doing some learning on the fly. Another challenge working with those kinds of models is the need for target data to learn on. After each booking, information about its fraud state needs to be provided. A challenge here is not only the provision of the data itself, but also the implementation of the deployed model. Unlike usual ml-models, online learning models have a changing state. They are not simple binaries and have to be kept in memory permanently, unlike e.g. scikit-learn models. Due to the duplex communication of the model with the other systems, the API of the model is also more complicated and cannot be implemented as easily in services such as SageMaker as it is the case with batch-learned models.

## Advantages 🤩
Enough with the pessimism. <span style="color:pink">Unifly</span> has a lot of smart people, and they didn't choose such a model for nothing. Especially for systems where data can change quickly, i.e. where so-called data drift occurs, an online learning model can react faster to these changes. More. <span style="color:pink">Unifly</span> can use this technology to train models for which no or too little data is currently available.

That all sounds pretty awesome! It not only has the potential to offer models with better performance. It also offers <span style="color:pink">Unifly</span> additional areas of application for machine learning models where a lack of data made this impossible until now. I bet some ML guy will design a small prototype in the coming weeks to test the potential of this technology. 😉

Thanks for reading! 🙏 Stay safe and keep predicting!
