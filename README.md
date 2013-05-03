bayes
=====

**Bayes** is a small Python class to reason about probabilities.
It uses a Bayesian system to extract features, crunch belief updates and
spew likelihoods back.

`b = Bayes([.5, .5])`: creates a new scenario with two equally likely classes

`b.update([.9, .1])`: updates the scenario with an event that is 9 times more
likely to have happened in the first class

`print(b.most_likely())`: prints which of the two classes is now more likely
  

Example Usage
-------------

```python
print ' -- Spam Filter --'
# Database with number of sightings of each words in (genuine, spam)
# emails.
words_odds = {'buy': (5, 100), 'viagra': (1, 1000), 'meeting': (15, 2)}
# Emails to be analyzed.
emails = [
          "let's schedule a meeting for tomorrow", # 100% genuine (meeting)
          "buy some viagra", # 100% spam (buy, viagra)
          "buy coffee for the meeting", # buy x meeting, should be genuine
         ]

for email in emails:
    # Start with priors of 90% chance being genuine, 10% spam.
    # Probabilities are normalized automatically.
    b = Bayes([('genuine', 90), ('spam', 10)])
    # Update probabilities, using the words in the emails as events and the
    # database of chances to figure out the change.
    b.update_from_events(email.split(), words_odds)
    # Print the email and if it's likely spam or not.
    print email[:15] + '...', b.most_likely()
    
print ''

print ' -- Spam Filter With Email Corpus -- '

# Email corpus. A hundred spam emails to buy products and with the word
# "meeting" thrown around. Genuine emails are about meetings and buying
# milk.
instances = {'spam': ["buy viagra", "buy cialis"] * 100 + ["meeting love"],
             'genuine': ["meeting tomorrow", "buy milk"] * 100}

# Use str.split to extract features/events/words from the corpus and build
# the model.
model = Bayes.extract_events_odds(instances, str.split)
# Create a new Bayes instance with 10%/90% priors on emails being genuine.
b = Bayes({'spam': .9, 'genuine': .1})
# Update beliefs with features/events/words from an email.
b.update_from_events("buy coffee for meeting".split(), model)
# Print the email and if it's likely spam or not.
print "'buy coffee for meeting'", ':', b

print ''

print ' -- Classic Cancer Test Problem --'
# 1% chance of having cancer.
b = Bayes([('not cancer', 0.99), ('cancer', 0.01)])
# Test positive, 9.6% false positives and 80% true positives
b.update((9.6, 80))
print b
print 'Most likely:', b.most_likely()

print ''

print ' -- Are You Cheating? -- '
results = ['heads', 'heads', 'tails', 'heads', 'heads']
events_odds = {'heads': {'honest': .5, 'cheating': .9},
               'tails': {'honest': .5, 'cheating': .1}}
b = Bayes({'cheating': .5, 'honest': .5})
b.update_from_events(results, events_odds)
print b


def b():
    return Bayes((0.99, 0.01), labels=['not cancer', 'cancer'])

# Random equivalent examples, all achieve the same result.
b() * (9.6, 80)
(b() * (9.6, 80)).opposite().opposite()
b().update({'not cancer': 9.6, 'cancer': 80})
b().update((9.6, 80))
b().update_from_events(['pos'], {'pos': (9.6, 80)})
b().update_from_tests([True], [(9.6, 80)])
Bayes([('not cancer', 0.99), ('cancer', 0.01)]) * (9.6, 80)
Bayes({'not cancer': 0.99, 'cancer': 0.01}) * {'not cancer': 9.6,
                                               'cancer': 80}
```
