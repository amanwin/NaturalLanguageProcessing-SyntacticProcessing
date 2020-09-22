# Syntactic Processing

## Introduction
**In this module**, you will learn algorithms and techniques used to analyse the syntax or the grammatical structure of sentences. In the first session, you will learn the basics of grammar (part-of-speech tags etc.) and write your own algorithms such as **HMMs(Hidden Markov Models) to build POS taggers**. In the second session, you will study algorithms to **parse the grammatical structure** of sentences such as CFGs, PCFGs and dependency parsing. Finally, in the third session, you will learn to build an **Information Extraction (IE)** system to parse flight booking queries for users using techniques such as **Named Entity Recognition (NER)**. You will also study a class of models called **Conditional Random Fields (CRFs)** which are widely used for building NER systems.

All these techniques fall under what is called **syntactic processing**.  

Syntactic processing is widely used in applications such as question answering systems, information extraction, sentiment analysis, grammar checking etc.

### The What and Why of Syntactic Processing
Let’s start with an example to understand **Syntactic Processing**:
* Canberra is the capital of Australia.
* Is Canberra the of Australia capital.

Both sentences have the same set of words, but only the first one is syntactically correct and comprehensible. Basic lexical processing techniques wouldn't be able to tell this difference. Therefore, more sophisticated syntactic processing techniques are required to understand the relationship between individual words in the sentence.

![title](img/syntaxanalysis.JPG)

Lexical analysis is data pre-processing and feature extraction step. It involves the analysis at word level. Syntactical analysis aims at finding structural relationships among the words of a sentence.

We’ve learnt about lexical processing in the last module. Lexical analysis aims at data cleaning and feature extraction, which it does by using techniques such as lemmatization, removing stopwords, rectifying misspelt words, etc. But, in syntactic analysis, our aim will be to understand the roles played by the words in the sentence, the relationship between words and to parse the grammatical structure of sentences. 

![title](img/whysyntaxanalysis.png)

Now that we understand the basic idea of syntactic processing, let's study the different levels of syntactic analysis.

### Parsing
A key task in syntactical processing is **parsing**. It means to break down a given sentence into its 'grammatical constituents'. Parsing is an important step in many applications which helps us better understand the linguistic structure of sentences.

Let’s understand parsing through an example. Let's say you ask a question answering (QA) system, such as Amazon's Alexa or Apple's Siri, the following question: "Who won <u>the cricket world cup in 2015</u>?"

The QA system can respond meaningfully only if it can understand that the phrase <u>'cricket world cup'</u> is related to the phrase 'in 2015'. The phrase <u>'in 2015'</u> refers to a specific time frame, and thus modifies the question significantly. Finding such dependencies or relations between the phrases of a sentence can be achieved using parsing techniques.

Let's take another example sentence to understand how a parsed sentence looks like: "The quick brown fox jumps over the table". The figure given below shows the three main constituents of this sentence. Note that actual parse trees are different from the simplified representation below.

![title](img/parsing.JPG)

This structure divides the sentence into three main constituents:
* 'The quick brown fox' is a noun phrase 
* 'jumps' is a verb phrase
* 'over the table' is a prepositional phrase.

![title](img/levelsofsyntax.png)

To summarise, you will study the following levels of syntactic analysis in this module:
* Part-of-speech (POS) tagging
* Constituency parsing
* Dependency parsing

Let’s understand the levels of syntax analysis using an example sentence: "The little boy went to the park."

**POS tagging** is the task of assigning a part of speech tag (POS tag) to each word. The POS tags identify the linguistic role of the word in the sentence. The POS tags of the sentence are:

![title](img/pos_tagging.JPG)

**Constituency parsers** divide the sentence into constituent phrases such as noun phrase, verb phrase, prepositional phrase etc. Each constituent phrase can itself be divided into further phrases. The constituency parse tree given below divides the sentence into two main phrases - a noun phrase and a verb phrase. The verb phrase is further divided into a verb and a prepositional phrase, and so on.

![title](img/constituency.JPG)

**Dependency Parsers** do not divide a sentence into constituent phrases, but rather establish relationships directly between the words themselves. The figure below is an example of a dependency parse tree of the sentence given above (generated using the spaCy dependency visualiser(https://explosion.ai/demos/displacy?text=The%20little%20boy%20went%20to%20the%20park&model=en_core_web_sm&cpu=0&cph=0)). In this module, you’ll understand when dependency parsing is more useful than constituency parsing and study the elements of dependency grammar.

![title](img/dependencyparse.JPG)

We will study these parsing techniques in the sections that follow. In the next few segments, we will study POS tagging in detail.

### Parts-of-Speech
Let’s start with the first level of syntactic analysis- POS (parts-of-speech) tagging. A word can be tagged as a noun, verb, adjective, adverb, preposition etc. depending upon its role in the sentence. Assigning the correct tag such as noun, verb, adjective etc. is one of the most fundamental tasks in syntactic analysis.  

Let’s say you ask your smart home device a question - "Ok Google, where can I get the permit to work in Australia?". Now, the word 'permit' can potentially have two POS tags - noun and a verb. In the phrase 'I need a work permit', the correct tag of 'permit' is 'noun'. On the other hand, in the phrase "Please permit me to take the exam.", the word 'permit' is a 'verb'.

Assigning the correct POS tags helps us better understand the intended meaning of a phrase or a sentence and is thus a crucial part of syntactic processing. In fact, all the subsequent parsing techniques (constituency parsing, dependency parsing etc.) use the part-of-speech tags to parse the sentence. 

![title](img/pos_tagging1.png)

![title](img/postags.png)

**Note**: You do not need to remember all the POS tags except for a few which are listed later on this page. You’ll pick up most of these tags as you work on the problems in the coming segments, but it’s important to be aware of all the types of tags. Now, let’s look at some other tags.

![title](img/postags2.JPG)

![title](img/listofpos.png)

There are 36 POS tags in the Penn treebank in NLTK. The file given below enlists the most commonly used POS tags. It is recommended to remember these common tags. It’ll help you avoid the trouble of looking up meanings of tags in the upcoming segments, so you can focus on the core concepts.  

[POS tags](dataset/POS_tags.pdf)

Note that the set of POS tags is not standard - some books/applications may use only the base forms such as NN, VB, JJ etc without using granular forms, though NLTK uses this set of tags(https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). Since we'll use NLTK heavily, we recommend you to read through this list of tags at least once.
