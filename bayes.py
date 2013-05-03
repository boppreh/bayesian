from collections import defaultdict
import os

def classify(instance, classes_instances, extractor=str.split):
    """
    Using `classes_instances` as supervised learning, classify `instance` into
    one of the example classes. `extractor` is a function to convert instances
    into a list of events/features to be analyzed, which defaults to a simple
    word extraction.
    """
    model = Bayes.extract_events_odds(classes_instances, extractor)
    b = Bayes({class_: 1.0 for class_ in classes_instances})
    b.update_from_events(extractor(instance), model)
    return b.most_likely()

def classify_file(file_, folders, extractor=str.split):
    """
    Classify `file_` into one of `folders`, based on the contents of the files
    already there.  `extractor` is a function to convert file contents
    into a list of events/features to be analyzed, which defaults to a simple
    word extraction.
    """
    classes_instances = defaultdict(list)
    for folder in folders:
        for child in os.listdir(folder):
            classes_instances[folder].append(os.path.join(folder, child))

    new_extractor = lambda f: extractor(open(f).read())
    return classify(file_, classes_instances, new_extractor)


class Bayes(list):
    """
    Class for Bayesian probabilistic evaluation through creation and update of
    beliefs. This is meant for abstract reasoning, not just classification.
    """
    @staticmethod
    def extract_events_odds(classes_instances, event_extractor=str.split):
        """
        Runs function `event_extractor` for every instance in every class in
        `classes_instances` ({class: [instances]}) and returns the odds of each
        event happening for each class.

        The result of this function is meant to be used in a future
        `update_from_events` call.
        """
        small = 0.000001
        events_odds = defaultdict(lambda: defaultdict(lambda: small))
        for class_, instances in classes_instances.items():
            for instance in instances:
                for event in event_extractor(instance):
                    events_odds[event][class_] += 1

        return events_odds

    def __init__(self, value=None, labels=None):
        """
        Creates a new Bayesian belief system.

        `value` can be another Bayes
        object to be copied, an array of odds, an array of (label, odds)
        tuples or a dictionary {label: odds}.

        `labels` is a list of names for the odds in `value`. Labels default to
        the their indexes.
        """
        if value is not None:
            # Convert dictionary.
            if isinstance(value, dict):
                labels = labels or sorted(value.keys())
                value = [value[label] for label in labels]

            # Convert list of tuples.
            elif labels is None and len(value) and isinstance(value[0], tuple):
                labels, value = zip(*value)

        super(Bayes, self).__init__(value)
        self.labels = labels or map(str, range(len(self)))

    def __getitem__(self, i):
        """ Returns the odds at index or label `i`. """
        if isinstance(i, str):
            return self[self.labels.index(i)]
        else:
            return super(Bayes, self).__getitem__(i)

    def __setitem__(self, i, value):
        """ Sets the odds at index or label `i`. """
        if isinstance(i, str):
            self[self.labels.index(i)] = value
        else:
            super(Bayes, self).__setitem__(i, value)

    def _cast(self, other):
        """
        Converts and unknown object into a Bayes object, keeping the same
        labels if possible.
        """
        if isinstance(other, Bayes):
            return other
        else:
            return Bayes(other, self.labels)

    def opposite(self):
        """
        Returns the opposite probabilities.
        Ex: [.7, .3] -> [.3, .7]
        """
        if 0 in self:
            return self._cast(1 if i == 0 else 0 for i in self)
        else:
            return self._cast(1 / i for i in self)

    def normalized(self):
        """
        Converts the list of odds into a list probabilities that sum to 1.
        """
        total = float(sum(self))
        return self._cast(i / total for i in self)

    def __mul__(self, other):
        """
        Creates a new instance with odds from both this and the other instance.
        Ex: [.5, .5] * [.9, .1] -> [.45, .05] (non normalized)
        """
        return self._cast(i * j for i, j in zip(self, self._cast(other)))

    def __div__(self, other):
        """
        Creates a new instance with odds from this instance and the opposite of
        the other.
        Ex: [.5, .5] / [.9, .1] -> [.555, 50.0] (non normalized)
        """
        return self * self._cast(other).opposite()

    def update(self, event):
        """
        Updates all current odds based on the likelihood of odds in event.
        Modifies the instance and returns itself.
        Ex: [.5, .5].update([.9, .1]) becomes [.45, .05] (non normalized)
        """
        self[:] = (self * self._cast(event)).normalized()
        return self

    def update_from_events(self, events, events_odds):
        """
        Perform an update for every event in events, taking the new odds from
        the dictionary events_odds (if available).
        Ex: [.5, .5].update_from_events(['pos'], {'pos': [.9, .1]})
        becomes [.45, .05] (non normalized)
        """
        for event in events:
            if event in events_odds:
                self.update(events_odds[event])
        return self

    def update_from_tests(self, tests_results, odds):
        """
        For every binary test in `tests_results`, updates the current belief
        using the item at the same position in `odds`. If the test is true, use
        the odds as is. If it's false, use it's opposite.
        Ex: [.5, .5].update_from_tests([True], [.9, .1]) becomes [.45, .05]
        (non normalized)
        """
        for result, chance in zip(tests_results, odds):
            if result:
                self.update(chance)
            else:
                self.update(self._cast(chance).opposite())
        return self

    def most_likely(self, cutoff=0.0):
        """
        Returns the label with most probability, or None if its probability is
        under `cutoff`.
        Ex: {a: .4, b: .6}.most_likely() -> b
            {a: .4, b: .6}.most_likely(cutoff=.7) -> None
        """
        normalized = self.normalized()
        max_value = max(normalized)

        if max_value > cutoff:
            return self.labels[normalized.index(max_value)]
        else:
            return None

    def is_likely(self, label, minimum_probability=0.5):
        """
        Returns if `label` has at least probability `minimum_probability`.
        Ex: {a: .4, b: .6}.is_likely(b) -> True
        """
        return self.normalized()[label] > minimum_probability

    def __str__(self):
        items = []
        for label, item in zip(self.labels, self.normalized()):
            items.append('{}: {}%'.format(label, round(item * 100, 2)))
        return 'Bayes({})'.format(', '.join(items))


if __name__ == '__main__':
    print ' -- Classic Cancer Test Problem --'
    # 1% chance of having cancer.
    b = Bayes([('not cancer', 0.99), ('cancer', 0.01)])
    # Test positive, 9.6% false positives and 80% true positives
    b.update((9.6, 80))
    print b
    print 'Most likely:', b.most_likely()

    print ''
    
    print ' -- Spam Filter With Existing Model --'
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

    print ' -- Are You Cheating? -- '
    results = ['heads', 'heads', 'tails', 'heads', 'heads']
    events_odds = {'heads': {'honest': .5, 'cheating': .9},
                   'tails': {'honest': .5, 'cheating': .1}}
    b = Bayes({'cheating': .5, 'honest': .5})
    b.update_from_events(results, events_odds)
    print b

    def b():
        return Bayes((0.99, 0.01), labels=['not cancer', 'cancer'])

    # Random equivalent examples
    b() * (9.6, 80)
    (b() * (9.6, 80)).opposite().opposite()
    b().update({'not cancer': 9.6, 'cancer': 80})
    b().update((9.6, 80))
    b().update_from_events(['pos'], {'pos': (9.6, 80)})
    b().update_from_tests([True], [(9.6, 80)])
    Bayes([('not cancer', 0.99), ('cancer', 0.01)]) * (9.6, 80)
    Bayes({'not cancer': 0.99, 'cancer': 0.01}) * {'not cancer': 9.6,
                                                   'cancer': 80}

