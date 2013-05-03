from collections import OrderedDict

class Bayes(OrderedDict):
    """
    Class for Bayesian probabilistic evaluation through creation and update of
    beliefs.
    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new instance from a dictionary {scenario: probability}, a
        list of tuples [(scenario, probability)], a list of probabilities for
        unnamed scenarios, or another Bayes object to be copied.

        Odds are normalized to [0, 1] probabilities automatically.
        """
        if len(args) == 1 and not isinstance(args[0], dict):
            try:
                args = [dict(args[0])]
            except TypeError:
                args = [dict(enumerate(args[0]))]

        super(Bayes, self).__init__(*args, **kwargs)
        self.normalize()

    def __mul__(self, other):
        copy = Bayes(self)
        copy.update([other])
        return copy

    def __div__(self, other):
        copy = Bayes(other)
        copy.negate()
        copy.update([self])
        return copy

    def negate(self):
        if 0 in self.values():
            for name, value in self.items():
                self[name] = 1 if value == 0 else 0
        else:
            for name, value in self.items():
                self[name] = 1 / value

        self.normalize()

    def update(self, event, do_normalize=True):
        try:
            for name, value in event.items():
                self[name] *= value
        except AttributeError:
            for name, value in zip(self, event):
                self[name] *= value

        if do_normalize:
            self.normalize()

    def update_from_events(self, events, events_odds):
        for event in events:
            if event in events_odds:
                self.update(events_odds[event])

    def update_from_tests(self, tests_results, tests_odds):
        for result, chance in zip(tests_results, tests_odds):
            if result:
                self.update(chance)
            else:
                b = Bayes(chance)
                b.negate()
                self.update(b)

    def most_likely(self, cutoff=0.0):
        values = self.values()
        max_value = max(values)

        if max_value > cutoff:
            return self.keys()[values.index(max_value)]
        else:
            return None

    def normalize(self):
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

    def __str__(self):
        pairs = []
        for key, value in self.items():
            pairs.append(key + ': ' + str(value * 100)[:5] + '%')
        return 'Bayes({})'.format(', '.join(pairs))

if __name__ == '__main__':
    print ' -- Cancer Test --'
    # 1% chance of having cancer.
    b = Bayes([('not cancer', 0.99), ('cancer', 0.01)])
    # Test positive, 9.6% false positives and 80% true positives
    b.update((9.6, 80))
    print b
    print 'Most likely:', b.most_likely()

    print ''
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
        # Print the email and if it's likely spam o rnot.
        print email[:15] + '...', b.most_likely()
