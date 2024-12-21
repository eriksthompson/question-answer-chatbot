# question-answer-chatbot
Version 1: Distilled version of BERT used to ask question and give answers given context paragraph with clues.

Version 2: Question processed by Spacy NLP to get nouns, proper nouns, subject, object. These are searched into wikipedia
and then sentences of wiki are ranked top 20 with msmarco assymmetric search. Finally, these top 20 are passed as context with question
into distilled bert to find answer. Not very efficient or accurate due to wrong wikipedia pages found often and irrelevant sentences ranked.

Version 3: Full question passed into wikipedia search engine and top 3 pages results parsed. Next, rank paragraphs on relevancy to question.
Finally, pass top 3 paragraphs into qa bert to answer question with context.
