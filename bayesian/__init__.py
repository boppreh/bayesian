from collections import defaultdict
import os

def classify(instance, classes_instances, extractor=str.split, priors=None):
    """
    Using `classes_instances` as supervised learning, classify `instance` into
    one of the example classes. `extractor` is a function to convert instances
    into a list of events/features to be analyzed, which defaults to a simple
    word extraction.
    """
    priors = priors or {class_: 1.0 for class_ in classes_instances}
    model = Bayes.extract_events_odds(classes_instances, extractor)
    b = Bayes(priors)
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
            child_path = os.path.join(folder, child)
            if os.path.isfile(child_path):
                classes_instances[folder].append(child_path)

    new_extractor = lambda f: extractor(open(f).read())
    return classify(file_, classes_instances, new_extractor)

def classify_folder(folder, extractor=str.split):
    """
    Move every file in `folder` into one of its subfolders, based on the
    contents of the files in those subfolders. `extractor` is a function to
    convert file contents into a list of events/features to be analyzed, which
    defaults to a simple word extraction.
    """
    subfolders = []
    files = []
    for item in os.listdir(folder):
        path = os.path.join(folder, item)
        if os.path.isdir(path):
            subfolders.append(path)
        else:
            files.append(path)

    for file_ in files:
        classification = classify_file(file_, subfolders, extractor)
        new_path = os.path.join(classification, os.path.basename(file_))
        if not os.path.exists(new_path):
            print(file_, classification)
            os.rename(file_, new_path)


from math import sqrt, pi, exp
def gaussian_distribution(values):
    """
    Given a list of values, returns the (mean, variance) tuple for a
    Gaussian (normal) distribution.
    """
    n = float(len(values))
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return (mean, variance)

def gaussian_probability(sample, distribution):
    """
    Given a sample value and the (mean, variance) distribution,
    return the probability of this sample belonging to the
    distribution.
    """
    mean, variance = distribution

    # Special case of degenerate distribution.
    if variance == 0:
        # 100% if sample is exactly at mean, otherwise 0%.
        return 0 if sample != mean else 1

    return (exp((sample - mean) ** 2 / (-2 * variance))
            / sqrt(2 * pi * variance))

def properties_distributions(classes_population):
    """
    Converts classes populations into classes distributions by property.
    {class: [{property: value}]} -> {property: {class: distribution}}
    """
    distributions = defaultdict(dict)
    for class_, population in classes_population.items():
        properties_instances = defaultdict(list)
        for properties in population:
            for property, value in properties.items():
                properties_instances[property].append(value)
        for property, instances in properties_instances.items():
            distributions[property][class_] = gaussian_distribution(instances)
    return distributions

def classify_normal(instance, classes_instances, priors=None):
    """
    Classify `instance` into one of the classes from `classes_instances`,
    calculating the probabilities from the Gaussian (normal) distribution
    from each classes' instances, starting from `priors` (uniform if not
    specified).

    classes_instances must be of type {class: [{property: value}]}
    priors must be of type {class: odds} and is automatically normalized.
    """
    priors = priors or {class_: 1.0 for class_ in classes_instances}
    b = Bayes(priors)

    distributions = properties_distributions(classes_instances)
    for property, value in instance.items():
        classes_distributions = distributions[property]
        probability_by_class = {class_: gaussian_probability(value, distribution)
                for class_, distribution in classes_distributions.items()}
        b.update(probability_by_class)

    return b.most_likely()
        
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
        if value is None:
            raise ValueError('Expected non-None value, got {}.'.format(value))

        if isinstance(value, dict):
            # Convert dictionary.
            labels = labels or list(sorted(value.keys()))
            raw_values = [value[label] for label in labels]
        else:
            value = list(value)
            if len(value) and isinstance(value[0], tuple):
                # Convert list of tuples.
                labels, raw_values = zip(*value)
            else:
                # Convert raw list of values.
                labels = [str(i) for i in range(len(value))]
                raw_values = value

        if len(labels) != len(set(labels)):
            raise ValueError('Labels must not be duplicated. Got {}.'.format(labels))

        self.labels = labels
        super(Bayes, self).__init__(raw_values)

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

    def __truediv__(self, other):
        """
        Creates a new instance with odds from this instance and the opposite of
        the other.
        Ex: [.5, .5] / [.9, .1] -> [.555, 5.0] (non normalized)
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
        depending on `odds`. If the test was True, use the odds as-is. If the
        test was false, use the opposite odds.
        Ex: [.5, .5].update_from_tests([True], [.9, .1]) becomes [.45, .05]
        (non normalized)
        """
        opposite_odds = self._cast(odds).opposite()
        for result in tests_results:
            if result:
                self.update(odds)
            else:
                self.update(opposite_odds)
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

    def __repr__(self):
        items = []
        for label, item in zip(self.labels, self.normalized()):
            items.append('{}: {}%'.format(label, round(item * 100, 2)))
        return 'Bayes({})'.format(', '.join(items))

    def __eq__(self, other):
        if isinstance(other, Bayes) and self.labels != other.labels:
            return False
        return list(self.normalized()) == list(self._cast(other).normalized())


if __name__ == '__main__':
    import sys
    for folder in sys.argv[1:]:
        classify_folder(folder)
