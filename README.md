# Identifying Reddit users based on text style analysis

This is currently a work in progress. Feel free to glance around, but in the meantime, please also take a look at some of my other projects. 


## Context

For centuries, authors have been able to write anonymously with the notion that their true identity would never be uncovered. That has all changed in the past few years, however, as machine learning methods have improved the efficacy of stylometry. Stylometry is the study of literary style, and involves identifying an author's writing style by uncovering unique patterns in word choice, sentence structure, punctuation, and more.

Some of the most famous examples of stylometric analysis include its use in identifying James Madison and Alexander Hamilton as the authors of the anonymously written Federalist Papers, successfully tying J.K. Rowling back to the anonymous author of the book, "A Cuckoo's Calling", and determining which Shakespearean plays Shakespeare actually wrote, which ones he co-wrote, and which ones were written by an entirely different person.

But these are all examples of large bodies of text written by professionals. Can stylometry still be effective in an informal setting with a smaller sample of text, such as social media? How few words do we need before we can start to distinguish an author's writing style from another's, and how many authors can we compare a body of anonymously written text to before we start to see two users with indistinguishably similar writing styles?

## Implementation

For this project, I wanted to see if I could answer these questions by analyzing users on Reddit. All users on the site submit posts and comments anonymously under a given username, and everyone's comments are publicly accessible. The only information needed to create a Reddit account is an email, so some users have been known to create multiple accounts. Perhaps, if I'm lucky, I can identify two or more accounts that belong to the same user. 

Because this is an unsupervised learning problem, I needed some way to validate the accuracy of my model. To do this, I took a user's entire comment history and randomly pulled out half of their comments, creating a new pseudo-user with these comments. Then I measured my model's success in being able to correctly match this pseudo-user back to the original user those comments were pulled from out of a pool of n users. 

### Baseline

One of the oldest techniques in stylometry dates back to the 1800's, where authors were compared by using the frequencies at which they use words of different lengths. Some tend to use short two and three-letter words more often, while others tend to pull from a larger vocabulary. The technique of using frequencies of these word lengths is commonly called Mendenhall's Characteristic Curves of Composition (MCCC). While fairly crude, this technique can still be used to identify authorship. We can determine whose curve of composition
most closely resembles the curve of the anonymous text by determining the curve with the lowest average RMSE. MCCC was first tested on a small group of 7 randomly chosen Redditors.

![Pseudo-users](word_lengths.png)

By iterating through each pseudo-user in Subset 2 and comparing it to each of the users in Subset 1, MCCC was able to correctly identify 5 of the 7 users. While promising, this technique is too simplistic to be used at scale. 

![Errors](word_length_errors.png)
![Errores](word_errors_df.png)

###  


