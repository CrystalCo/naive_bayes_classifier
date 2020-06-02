# Crystal A. Contreras  Spring CSC-480  Assignment 4
import heapq
import numpy

class Naive_Bayesian_Classifier:
    def __init__(self, train_class_file, train_matrix_file, test_class_file, test_matrix_file, terms_file):
        self.actual_class_table = []
        self.conditional_probabilities = {}
        self.terms_file = terms_file
        self.test_class_file = test_class_file
        self.test_matrix_file = test_matrix_file
        self.train_class_file = train_class_file
        self.train_matrix_file = train_matrix_file

    def Naive_Bayesian_Classifier(self):
        """
            First, uses the Naive Bayesian algorithm to create a model in the form of probabilities.
            Then, tests the model by predicting the class label for a test data set.
            Lastly, measures the classification accuracy of the model.
        """
        ##### MODEL CONSTRUCTION #####
        predicted_classes = self.naive_bayesian_algorithm()

        ##### MODEL EVALUATION #####
        accuracy_metrics = self._evaluation(self.test_class_file, predicted_classes)
            
        # Prints the ratio of correct predictions to the number of test instances.
        print(f'\nClassification Accuracy: {accuracy_metrics}\n')

        # Option to output the predicted vs actual class labels for a portion of the test data.
        display_row_count = input("Enter the number of rows to display for the predicted vs actual class labels: ")
        self._print_actual_vs_predicted_class_labels(predicted_classes, display_row_count)

        # View the learned class probabilities for specified terms
        self._print_learned_class_probabilities_for_term()

    def _concat_documents(self, documents):
        total = 0
        concat_docs = {}
        for doc in documents.keys():
            for word in documents[doc].keys():
                total += documents[doc][word]
                if word not in concat_docs.keys():
                    concat_docs[word] = documents[doc][word]
                else:
                    concat_docs[word] = concat_docs[word] + documents[doc][word]
        return [total, concat_docs]

    def _evaluation(self, actual_class_documents, predicted_class_table):
        """
            Evaluates/meausures the accuracy of the classifier.
            Takes as input: 
                - the actual classes for the test data; and
                - our predicted classes for the test data
            Returns the Classification Accuracy (i.e. the ratio of correct predictions to number of test instances)
        """
        actual_class_table = self._get_file_by_line(actual_class_documents)

        # Assign variable to store count of correct predictions and false predictions for each class
        correct = 0
        incorrect = 0
        
        # For each actual class row
        for doc_row in range(len(actual_class_table)):
            actual_class_table[doc_row] = actual_class_table[doc_row].split('\t')
            # Compare the actual class value with the predicted class value for that document
            if int(actual_class_table[doc_row][1]) == predicted_class_table[doc_row][1]:
                correct += 1
            else:
                incorrect +=1

        self.actual_class_table = actual_class_table
        return correct/(correct + incorrect)

    def _get_classes(self, class_filepath):
        """
            Reads in a file, splits that file by rows as a list containing the 
            document index in the first column and class ID in the 2nd column.
            Returns a dictionary of distinct classes with empty sets as their values
            as placeholders for future term input.
        """
        classes_filepath = class_filepath
        class_list = self._get_file_by_line(classes_filepath)
        total_num_of_documents = len(class_list)

        class_dict = dict()
        
        for i in range(total_num_of_documents):
            # get class set by splitting lines by tab character
            index_class = class_list[i].split("\t")
            document_index = index_class[0]
            class_id = index_class[1]

            # if key in dictionary does not already exist, add to dictionary w/value of 1 (OPTIONAL: add to set too)
            doc_name = 'd' + document_index

            if class_id not in class_dict.keys():
                class_dict[class_id] = { doc_name: {} }
            else:
                class_dict[class_id][doc_name] = {}
        
        return class_dict

    def _get_data_vector(self, class_file, matrix_file, terms_file):
        """ 
            Object that returns class IDs (aka categories) 
            & their respective documents with counts for each word.
            Used for the training component of the Naive Bayes Algorithm
        """
        classes = self._get_classes(class_file)
        vocabulary = self._get_vocab_list(terms_file)
        train_matrix = self._get_file_by_line(matrix_file)
        docs_with_vocab_matrix = self._get_documents_with_vocab(vocabulary, train_matrix)

        for i in classes:
            for key in classes[i].keys():
                classes[i][key] = docs_with_vocab_matrix[key]

        data_vector = {
            'classes': classes,
            'vocabulary': vocabulary
        }

        return data_vector

    def _get_doc_count(self, classes):
        total_docs = 0

        for c in range(len(classes)):
            total_docs += len(classes[str(c)])

        return total_docs

    def _get_documents_with_vocab(self, sample_vocab, sample_matrix):
        sample_dict = {}
        split_matrix = []

        for i in sample_matrix:
            split = i.split('\t')
            doc_index = len(split)
            for j in range(doc_index):
                doc_name = 'd'+str(j)
                sample_dict[doc_name] = {}
            split_matrix.append(split)

        for v in range(len(sample_vocab)):
            key_count = 0
            for key in sample_dict.keys():
                vocab_name = sample_vocab[v]
                v_row = split_matrix[v]
                freq = v_row[key_count]
                sample_dict[key][vocab_name] = float(freq)
                key_count += 1

        return sample_dict

    def _get_file_by_line(self, filepath):
        """
            Input: a string pathname to a file
            Reads in the file by line. 
            Removes the newline character at the end of each line.
            Returns the file rows as a list of strings.
        """
        file = open(filepath)
        content_list = file.readlines()
        file.close()

        for i in range(len(content_list)):
            content_list[i] = content_list[i].rstrip("\n")

        if content_list[len(content_list)-1] == '':
            content_list.pop() # Remove last line 
        
        return content_list

    def _get_vocab_list(self, vocab_filepath):
        'Create list for vocabulary in documents'
        filepath = vocab_filepath   #alternatively, filepath = input("Enter the Terms file path: ")
        vocab = self._get_file_by_line(filepath)    
        return vocab

    def naive_bayesian_algorithm(self):
        """
            Split into 2 parts: training component & testing component.
                Training component returns a set of all priors for each category: P(c)
                and set of conditional probabilities for each term given category: P(term|c)
        """
        ##### TRAINING COMPONENT #####
        # Get data representation vector
        data_vector = self._get_data_vector(self.train_class_file, self.train_matrix_file, self.terms_file)

        # Get list of vocabulary/terms 
        vocab = data_vector['vocabulary']
        vocab_length = len(vocab)

        # Get dictionary of categories with their respective documents 
        categories = data_vector['classes']

        total_num_of_documents = self._get_doc_count(categories)
            
        for category in categories.keys():
            # Probability(category) = # of docs in that category divided by total docs
            p_c = len(categories[category]) / total_num_of_documents
            self.conditional_probabilities[category] = {
                'p_c': p_c
            }

            # Conditional Probability(E|category)
            docs = categories[category]                 # All docs in this category
            t_i = self._concat_documents(docs)          # Concatenated docs
            n_i = t_i[0]                                # Total # of word occurences in t_i
            total_word_frequencies_in_documents = t_i[1]    # Dictionary of vocab as keys and their total frequencies as values for this subset of docs
            
            # For each word in the vocab, calculate the # of occurences of word in docs
            for word in vocab:
                n_ij = total_word_frequencies_in_documents[word]
                p_word_given_category = (n_ij + 1) / (n_i + vocab_length)   # LaPlace smoothing
                self.conditional_probabilities[category][word] = p_word_given_category

        ##### TESTING/CLASSIFICATION COMPONENT #####
        predicted_classes = self._testing_component(vocab, self.test_matrix_file)
        return predicted_classes

    def _predict_category_for_doc(self, document):
        """ 
            Takes as input one document.
            Returns a tuple with the probability of category given document in the first index
            and the predicted class label in the 2nd index.
        """
        # Let n be the # of word occurrences in X 
        n = 0
        # Condense dictionary to only vocab terms being used
        condensed_dict = {} 

        for vocab in document.keys():
            if document[vocab] > 0:
                n += document[vocab]
                condensed_dict[vocab] = document[vocab]

        p_category_given_document_numerator = []
        p_category_given_document_denominator = 0
        class_prediction = []   # will use heapq.  Remember to * -1 to return max up top

        # Return the category
        for category in self.conditional_probabilities.keys():
            # Start the array with the Probability of the category.  P(ci)
            p_category_given_doc = [self.conditional_probabilities[category]['p_c']]
            # P(ai|ci)*P(ai|ci)*P(ai|ci)*P(ai|ci)
            for term in condensed_dict.keys():
                # get the conditional probability for the terms that appear in the doc, and make its number of occurences its exponent. 
                p_a_given_c = self.conditional_probabilities[category][term]
                p_a_given_c = p_a_given_c**condensed_dict[term]
                # Append the result to our P(category | doc) array to multiply later
                p_category_given_doc.append(p_a_given_c)
            
            # Multiply P(ci)*P(ai|ci)*P(ai|ci)*P(ai|ci)*P(ai|ci)
            result = numpy.prod(p_category_given_doc)
            # Append result to P(c|doc) numerator
            p_category_given_document_numerator.append(result)
            # Add result to the denominator to divide by later 
            p_category_given_document_denominator += result
            
        # Get category prediction for this document by dividing the P(c|d) by the sum of the numerators
        for p_cat_given_doc in p_category_given_document_numerator:
            result = p_cat_given_doc / p_category_given_document_denominator
            # Append to heapq.  Multiply by -1 to return the max Probability of class given document from the heap
            heapq.heappush(class_prediction, (result * -1, p_category_given_document_numerator.index(p_cat_given_doc)))

        category = heapq.heappop(class_prediction)
        p_c_given_d = -(category[0])    # Turn it back into a positive int

        # returns probability of class given document and the predicted class label
        return (p_c_given_d, category[1])

    def _print_actual_vs_predicted_class_labels(self, predicted_classes, total_rows):
        """ Takes as input the number of rows you'd like to see the predicted classes for.
        """
        print("\nDocument Index   Predicted Class   Actual Class")
        total_rows = int(total_rows)

        for i in range(total_rows):
            print(f'{predicted_classes[i][0]}\t{predicted_classes[i][1]}\t{self.actual_class_table[i][1]}')
        else:
            pass
        print()

    def _print_learned_class_probabilities_for_term(self):
        """ 
            Returns a tuple with the class label in the first position
            and the probability for the learned class of a term in the 2nd position.
        """
        term = ''
        while term != 'end()':
            term = input("Enter the term for which you want to see the class probability of, or enter 'end()' to quit: ")
            if term == 'end()':
                break
            else:
                for key in self.conditional_probabilities.keys():
                    print('Term: {}.  Class label: {}.  Probability: {}'.format(term, key, self.conditional_probabilities[key][term]))

    def _testing_component(self, vocabulary, test_matrix_file):
        """
            Takes as input a vocabulary list and a pathname to a test matrix.
            Returns a list of arrays of length 3 with the document index name in the first column,
            predicted class label in the 2nd column, and probability of class/category given document
            in the third column.
        """
        test_matrix = self._get_file_by_line(test_matrix_file)
        docs_with_vocab_matrix = self._get_documents_with_vocab(vocabulary, test_matrix)
        classified_documents = []

        for document in docs_with_vocab_matrix.keys():
            prediction = self._predict_category_for_doc(docs_with_vocab_matrix[document])
            prob_cat_given_doc = prediction[0]
            class_label = prediction[1]
            classified_documents.append([document, class_label, prob_cat_given_doc])
        return classified_documents

train_class_file = './newsgroups/trainClasses.txt'
train_matrix_file = './newsgroups/trainMatrix.txt'
test_class_file = './newsgroups/testClasses.txt'
test_matrix_file = './newsgroups/testMatrix.txt'
terms_file = './newsgroups/terms.txt'

classifier = Naive_Bayesian_Classifier(train_class_file, train_matrix_file, test_class_file, test_matrix_file, terms_file)
classifier.Naive_Bayesian_Classifier()