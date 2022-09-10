import numpy
import sklearn.metrics
import common.utils
import common.numpy
import math
import numpy


class CleanEvaluation:
    """
    Evaluation on clean examples.
    """

    def __init__(self, probabilities, labels, validation=0.1):
        """
        Constructor.

        :param probabilities: predicted probabilities
        :type probabilities: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        :param validation: fraction of validation examples
        :type validation: float
        """

        # probabilities can be fewer than labels
        # the last validation% of labels should be for validation, even if probabilities contains more

        assert validation >= 0
        labels = numpy.squeeze(labels)
        assert len(labels.shape) == 1
        assert len(probabilities.shape) == 2
        assert probabilities.shape[0] == labels.shape[0]
        #if probabilities.shape[1] == 1:
        #    probabilities = numpy.concatenate((probabilities, 1 - probabilities), axis=1)
        assert probabilities.shape[1] == numpy.max(labels) + 1, (probabilities.shape[1], numpy.max(labels) + 1)
        marginals = numpy.sum(probabilities, axis=1)
        assert numpy.allclose(marginals, numpy.ones(marginals.shape)), marginals

        self.N = labels.shape[0]
        """ (int) Number of samples. """

        self.test_N = math.ceil(self.N*(1 - validation))
        """ (int) Test examples. """#

        if validation <= 0:
            assert self.test_N == self.N

        self.validation_N = self.N - self.test_N
        """ (int) Validation examples. """

        if validation <= 0:
            assert self.validation_N == 0

        self.test_probabilities = probabilities[:self.test_N]
        """ (numpy.ndarray) Test probabilities. """

        self.test_labels = labels[:self.test_N]
        """ (numpy.ndarray) Test labels. """

        self.test_predictions = numpy.argmax(self.test_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.test_errors = (self.test_predictions != self.test_labels)
        """ (numpy.ndarray) Test errors. """

        self.test_confidences = self.test_probabilities[numpy.arange(self.test_predictions.shape[0]), self.test_predictions]
        """ (numpy.ndarray) Test confidences. """

        self.validation_probabilities = None
        self.validation_confidences = None
        self.validation_predictions = None
        self.validation_errors = None
        self.validation_confidences = None

        if validation > 0:
            self.validation_probabilities = probabilities[self.test_N:]
            """ (numpy.ndarray) Validation probabilities. """

            self.validation_labels = labels[self.test_N:]
            """ (numpy.ndarray) Validation labels."""

            self.validation_predictions = numpy.argmax(self.validation_probabilities, axis=1)
            """ (numpy.ndarray) Validation predicted labels. """

            self.validation_errors = (self.validation_predictions != self.validation_labels)
            """ (numpy.ndarray) Validation errors. """

            self.validation_confidences = self.validation_probabilities[numpy.arange(self.validation_predictions.shape[0]), self.validation_predictions]
            """ (numpy.ndarray) Validation confidences. """

        # some cache variables
        self.sorted_correct_validation_confidences = None
        """ (numpy.ndarray) Caches sorted confidneces. """

    # this version rounds up
    # example: if 81% are requested, but only 5 samples are given, i.e. 0%, 20%, 40%, 60%, 80%, 100% are possible,
    # the confidence for 100% is returned
    def confidence_at_tpr(self, tpr):
        """
        Confidence threshold for given true positive rate.

        :param tpr: true positive rate in [0, 1]
        :type tpr: float
        :return: confidence threshold
        :rtype: float
        """

        assert self.validation_confidences is not None
        assert tpr > 0

        # true positives are correctly classified examples
        if self.sorted_correct_validation_confidences is None:
            correct_validation_confidences = self.validation_confidences[numpy.logical_not(self.validation_errors)]
            self.sorted_correct_validation_confidences = numpy.sort(numpy.copy(correct_validation_confidences))
        # rounding is a hack see tests
        cutoff = math.floor(self.sorted_correct_validation_confidences.shape[0] * round((1 - tpr), 2))
        assert cutoff >= 0
        assert cutoff < self.sorted_correct_validation_confidences.shape[0]
        return self.sorted_correct_validation_confidences[cutoff]

    def tpr_at_confidence(self, threshold):
        """
        True positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: true positive rate
        :rtype: float
        """

        return numpy.sum(self.test_confidences[numpy.logical_not(self.test_errors)] >= threshold) / float(numpy.sum(numpy.logical_not(self.test_errors)))

    def validation_tpr_at_confidence(self, threshold):
        """
        True positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: true positive rate
        :rtype: float
        """

        validation_confidences = self.validation_confidences[numpy.logical_not(self.validation_errors)]
        return numpy.sum(validation_confidences >= threshold) / float(validation_confidences.shape[0])

    def fpr_at_confidence(self, threshold):
        """
        False positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: false positive rate
        :rtype: float
        """

        return numpy.sum(self.test_confidences[self.test_errors] >= threshold) / float(numpy.sum(self.test_errors))

    def test_error(self):
        """
        Test error.

        :return: test error
        :rtype: float
        """

        return numpy.sum(self.test_errors.astype(int)) / float(self.test_N)

    def test_error_at_confidence(self, threshold):
        """
        Test error for given confidence threshold.

        :param threshold: confidence threshold
        :type threshold: float
        :return test error
        :rtype: float
        """

        nominator = numpy.sum(numpy.logical_and(self.test_errors, self.test_confidences >= threshold))
        denominator = numpy.sum(self.test_confidences >= threshold)
        if denominator > 0:
            return nominator / float(denominator)
        else:
            return 0

    def test_error_curve(self):
        """
        Test error for different confidence threshold.

        :return: test errors and confidences
        :rtype: numpy.ndarray, numpy.ndarray
        """

        scores = self.test_confidences
        sort = numpy.argsort(scores, axis=0)
        sorted_scores = scores[sort]

        test_errors = numpy.zeros((scores.shape[0]))
        thresholds = numpy.zeros((scores.shape[0]))

        for i in range(sort.shape[0]):
            thresholds[i] = sorted_scores[i]
            test_errors[i] = numpy.sum(self.test_errors[self.test_confidences >= thresholds[i]]) / float(numpy.sum(self.test_confidences >= thresholds[i]))

        return test_errors, thresholds

    def receiver_operating_characteristic_labels_scores(self):
        """
        Define labels and scores for ROC.

        :return: labels and scores for sklearn.metrics.roc_auc_score
        :rtype: numpy.ndarray, numpy.ndarray
        """

        return numpy.logical_not(self.test_errors).astype(int), self.test_confidences

    def receiver_operating_characteristic_auc(self):
        """
        Computes the ROC curve for correct classified vs. incorrect classified.

        :return: ROC AUC score
        :rtype: float
        """

        labels, scores = self.receiver_operating_characteristic_labels_scores()
        # what's the ROC AUC if there is only one class?
        if numpy.unique(labels).shape[0] == 1:
            return 1
        else:
            return sklearn.metrics.roc_auc_score(labels, scores)

    def receiver_operating_characteristic_curve(self):
        """
        Computes the ROC curve for correct classified vs. incorrect classified.

        :return: false positive rates, true positive rates, thresholds
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        labels, scores = self.receiver_operating_characteristic_labels_scores()
        return sklearn.metrics.roc_curve(labels, scores)

    def confidence_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.95)

    def confidence_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.98)

    def confidence_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.99)

    def confidence_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.995)

    def tpr_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.95))

    def tpr_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.98))

    def tpr_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.99))

    def tpr_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.995))

    def validation_tpr_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.95))

    def validation_tpr_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.98))

    def validation_tpr_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.99))

    def validation_tpr_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.995))

    def fpr_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.95))

    def fpr_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.98))

    def fpr_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.99))

    def fpr_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.995))

    def test_error_at_95tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.95))

    def test_error_at_98tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.98))

    def test_error_at_99tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.99))

    def test_error_at_995tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.995))