import numpy
import sklearn.metrics
import common.utils
import math
import numpy
from .clean_evaluation import *


class DistalEvaluation(CleanEvaluation):
    """
    Evaluation on adversarial and clean examples.
    """

    def __init__(self, clean_probabilities, distal_probabilities, labels, validation=0.1, errors=None, include_misclassifications=False):
        """
        Constructor.

        :param clean_probabilities: probabilities on clean examles
        :type clean_probabilities: numpy.ndarray
        :param distal_probabilities: probabilities on adversarial examples
        :type distal_probabilities: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray+
        :param validation: fraction of validation examples
        :type validation: float
        :param errors: errors to determine worst case
        :type errors: None or numpy.ndarray
        :param include_misclassifications: include mis classifications in confidence threshold computation
        :type include_misclassifications: bool
        """

        assert validation >= 0
        labels = numpy.squeeze(labels)
        assert len(labels.shape) == 1

        assert len(clean_probabilities.shape) == 2
        assert clean_probabilities.shape[0] == labels.shape[0]
        assert clean_probabilities.shape[1] == numpy.max(labels) + 1
        assert len(distal_probabilities.shape) == len(clean_probabilities.shape) + 1
        assert distal_probabilities.shape[2] == clean_probabilities.shape[1]
        assert distal_probabilities.shape[1] <= clean_probabilities.shape[0]
        if validation > 0:
            assert distal_probabilities.shape[1] + int(validation * clean_probabilities.shape[0]) <= clean_probabilities.shape[0]

        self.A = distal_probabilities.shape[0]
        """ (int) Attempts. """

        self.N = distal_probabilities.shape[1]
        """ (int) Samples. """

        if errors is not None:
            selected = numpy.argmin(errors, axis=0)
            distal_probabilities = distal_probabilities[selected, numpy.arange(distal_probabilities.shape[1])]
            reference_labels = labels[:self.N]
            reference_probabilities = clean_probabilities[:self.N]
            self.A = 1  # !
        else:
            distal_probabilities = distal_probabilities.reshape(self.A * self.N, -1)
            reference_labels = numpy.tile(labels[:self.N], self.A)
            reference_probabilities = numpy.tile(clean_probabilities[:self.N], (self.A, 1))

        assert distal_probabilities.shape[0] == reference_probabilities.shape[0]
        assert distal_probabilities.shape[1] == reference_probabilities.shape[1]
        assert len(distal_probabilities.shape) == len(reference_probabilities.shape)
        assert reference_probabilities.shape[0] == reference_labels.shape[0]

        distal_marginals = numpy.sum(distal_probabilities, axis=1)
        assert numpy.allclose(distal_marginals, numpy.ones(distal_marginals.shape))

        clean_marginals = numpy.sum(clean_probabilities, axis=1)
        assert numpy.allclose(clean_marginals, numpy.ones(clean_marginals.shape))

        distal_marginals = numpy.sum(distal_probabilities, axis=1)
        assert numpy.allclose(distal_marginals, numpy.ones(distal_marginals.shape))

        self.test_AN = self.A * self.N
        """ (int) Test examples. """

        self.validation_N = int((1 - validation) * self.N)
        """ (int) Validation examples. """

        self.test_probabilities = reference_probabilities
        """ (numpy.ndarray) Test probabilities. """

        self.test_labels = reference_labels
        """ (numpy.ndarray) Test labels. """

        self.test_predictions = numpy.argmax(self.test_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.test_errors = (self.test_predictions != self.test_labels)
        """ (numpy.ndarray) Test errors. """

        self.test_confidences = numpy.max(self.test_probabilities, axis=1)
        """ (numpy.ndarray) Test confidences. """

        self.validation_probabilities = None
        self.validation_confidences = None
        self.validation_predictions = None
        self.validation_errors = None
        self.validation_confidences = None

        if validation > 0:
            self.validation_probabilities = clean_probabilities[self.validation_N:]
            """ (numpy.ndarray) Validation probabilities. """

            self.validation_labels = labels[self.validation_N:]
            """ (numpy.ndarray) Validation labels."""

            self.validation_predictions = numpy.argmax(self.validation_probabilities, axis=1)
            """ (numpy.ndarray) Validation predicted labels. """

            self.validation_errors = (self.validation_predictions != self.validation_labels)
            """ (numpy.ndarray) Validation errors. """

            #self.validation_confidences = self.validation_probabilities[numpy.arange(self.validation_predictions.shape[0]), self.validation_predictions]
            self.validation_confidences = numpy.max(self.validation_probabilities, axis=1)
            """ (numpy.ndarray) Validation confidences. """

        self.test_distal_probabilities = distal_probabilities
        """ (numpy.ndarray) Test probabilities. """

        #self.test_distal_confidences = numpy.max(self.test_distal_probabilities, axis=1)
        self.test_distal_confidences = numpy.max(self.test_distal_probabilities, axis=1)
        """ (numpy.ndarray) Test confidences. """

        # NOTE: we do not need validation part of adversarial examples as threshold only depends on clean examples!

        # some cache variables
        self.sorted_validation_confidences = None
        """ (numpy.ndarray) Caches confidences. """

        self.include_misclassifications = include_misclassifications
        """ (bool) Include misclassifications. """

    def confidence_at_tpr(self, tpr):
        """
        Confidence threshold for given true positive rate.

        :param tpr: true positive rate in [0, 1]
        ;type tpr: float
        :return: confidence threshold
        :rtype: float
        """

        assert self.validation_confidences is not None
        assert tpr > 0

        # true positives are real examples
        if self.sorted_validation_confidences is None:
            if self.include_misclassifications:
                self.sorted_validation_confidences = numpy.sort(numpy.copy(self.validation_confidences))
            # This is computing the threshold only on correctly classified examples
            else:
                self.sorted_validation_confidences = numpy.sort(numpy.copy(self.validation_confidences[numpy.logical_not(self.validation_errors)]))
        # rounding is a hack see tests
        cutoff = math.floor(self.sorted_validation_confidences.shape[0] * round((1 - tpr), 2))
        assert cutoff >= 0
        assert cutoff < self.sorted_validation_confidences.shape[0]
        return self.sorted_validation_confidences[cutoff]

    def tpr_at_confidence(self, threshold):
        """
        True positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: false positive rate
        :rtype: float
        """

        return numpy.sum(self.test_confidences >= threshold) / float(self.test_AN)

    def validation_tpr_at_confidence(self, threshold):
        """
        True positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: true positive rate
        :rtype: float
        """

        if self.include_misclassifications:
            validation_confidences = self.validation_confidences
        else:
            validation_confidences = self.validation_confidences[numpy.logical_not(self.validation_errors)]

        return numpy.sum(validation_confidences >= threshold) / float(validation_confidences.shape[0])

    def test_error(self):
        """
        Test error.

        :return: test error
        :rtype: float
        """

        return numpy.sum(self.test_errors.astype(int)) / float(self.test_AN)

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

    def fpr_at_confidence(self, threshold):
        """
        False positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: false positive rate
        :rtype: float
        """

        return numpy.sum(self.test_distal_confidences >= threshold) / float(self.test_AN)

    def tnr_at_confidence(self, threshold):
        """
        False positive rate at confidence threshold.

        :param threshold: confidence threshold in [0, 1]
        :type threshold: float
        :return: false positive rate
        :rtype: float
        """

        return numpy.sum(self.test_distal_confidences < threshold) / float(self.test_AN)

    def receiver_operating_characteristic_labels_scores(self):
        """
        Labels and scores for ROC.

        :return: labels and scores for sklearn.metrics.roc_auc_score
        :rtype: numpy.ndarray, numpy.ndarray
        """

        labels = numpy.concatenate((
            numpy.zeros(self.test_AN),
            numpy.ones(self.test_AN),
        ))
        scores = numpy.concatenate((
            self.test_distal_confidences,
            self.test_confidences
        ))

        return labels, scores

    def receiver_operating_characteristic_auc(self):
        """
        Computes the ROC curve for correct classified vs. incorrect classified.

        :return: ROC AUC score
        :rtype: float
        """

        labels, scores = self.receiver_operating_characteristic_labels_scores()
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

    def confidence_at_90tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.confidence_at_tpr(0.9)

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

    def tpr_at_90tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tpr_at_confidence(self.confidence_at_tpr(0.9))

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

    def validation_tpr_at_90tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.validation_tpr_at_confidence(self.confidence_at_tpr(0.9))

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

    def fpr_at_90tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.fpr_at_confidence(self.confidence_at_tpr(0.9))

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

    def tnr_at_90tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.9))

    def tnr_at_95tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.95))

    def tnr_at_98tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.98))

    def tnr_at_99tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.99))

    def tnr_at_995tpr(self):
        """
        TNR at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.tnr_at_confidence(self.confidence_at_tpr(0.995))

    def test_error_at_90tpr(self):
        """
        Test error at 95%TPR.

        :return: robust test error
        :rtype: float
        """

        return self.test_error_at_confidence(self.confidence_at_tpr(0.9))

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