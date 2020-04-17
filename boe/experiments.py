import process_text
from preprocess_pickle import get_text_and_labels
from boe_classifier import train, test
import year_regressor


if __name__ == "__main__":

    print("WOTD")
    print("Processing WOTD files")
    wotd_train_with_ci, wotd_train_no_ci, wotd_train_labels = get_text_and_labels('data/train/wotd.pkl')
    wotd_test_with_ci, wotd_test_no_ci, wotd_test_labels = get_text_and_labels('data/test/wotd.pkl')

    print("Classification - no CI")
    clf = train(wotd_train_no_ci, wotd_train_labels)
    test(wotd_test_no_ci, wotd_test_labels, clf)

    print("Classification - with  CI")
    clf = train(wotd_train_with_ci, wotd_train_labels)
    test(wotd_test_with_ci, wotd_test_labels, clf)

    print("Regression - no CI")
    year_regressor.train_regression(wotd_train_no_ci, wotd_train_labels)
    year_regressor.test_regression(wotd_test_no_ci, wotd_test_labels)

    print("Regression - with CI")
    year_regressor.train_regression(wotd_train_with_ci, wotd_train_labels)
    year_regressor.test_regression(wotd_test_with_ci, wotd_test_labels)

    print("OTD")
    print("Processing OTD files")
    otd_train_with_ci, otd_train_no_ci, otd_train_labels = get_text_and_labels('data/train/otd2.pkl')
    otd_test_with_ci, otd_test_no_ci, otd_test_labels = get_text_and_labels('data/test/otd2.pkl')

    print("Classification - no CI")
    clf = train(otd_train_no_ci, otd_train_labels)
    test(otd_test_no_ci, otd_test_labels, clf)

    print("Classification - with CI")
    clf = train(otd_train_with_ci, otd_train_labels)
    test(otd_test_with_ci, otd_test_labels, clf)
