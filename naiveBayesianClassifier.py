# Crystal A. Contreras  Spring CSC-480  Assignment 4
import heapq
import numpy

class Naive_Bayesian_Classifier:
    def __init__(self, train_class_file, train_matrix_file, test_class_file, test_matrix_file, terms_file):
        self.conditional_probabilities = {}
        self.train_class_file = train_class_file
        self.train_matrix_file = train_matrix_file
        self.test_class_file = test_class_file
        self.test_matrix_file = test_matrix_file
        self.terms_file = terms_file

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
                p_word_given_category = (n_ij + 1) / (n_i + vocab_length)
                self.conditional_probabilities[category][word] = p_word_given_category

        print(f'Conditional Probabilities: {self.conditional_probabilities}')

        ##### TESTING COMPONENT #####
        test_one_doc_file = self._get_file_by_line('/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/test1doc.txt')
        self._nbc_testing_component(vocab, test_one_doc_file)

       
    def _nbc_testing_component(self, vocabulary, test_matrix_file):
        # Given a test document X
        docs_with_vocab_matrix = self._get_documents_with_vocab(vocabulary, test_matrix_file)
        print("Testing Matrix w/Vocab: ", docs_with_vocab_matrix)
        
        # Let n be the # of word occurrences in X 
        n = 0
        # Condense dictionary to only vocab terms being used
        condensed_dict = {} 

        for doc in docs_with_vocab_matrix.keys():
            for vocab in docs_with_vocab_matrix[doc].keys():
                if docs_with_vocab_matrix[doc][vocab] > 0:
                    n += docs_with_vocab_matrix[doc][vocab]
                    condensed_dict[vocab] = docs_with_vocab_matrix[doc][vocab]
        print("Condensed Dictionary: ", condensed_dict)

        p_category_given_document_numerator = []
        p_category_given_document_denominator = 0
        class_prediction = []   # will use heapq.  Remember to * -1 to return max up top

        # Return the category
        for category in self.conditional_probabilities.keys():
            # Start the array with the Probability of the category
            p_category_given_doc = [self.conditional_probabilities[category]['p_c']]
            
            for term in condensed_dict.keys():
                # get the conditional probability only for the terms that appear in the doc, and make its frequency its exponent
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

        for i in p_category_given_document_numerator:
            result = i / p_category_given_document_denominator
            print(f'Result: {result}')
            # Append to heapq.  Multiply by -1 to return max from the heap
            heapq.heappush(class_prediction, (result * -1, p_category_given_document_numerator.index(i)))

        category = heapq.heappop(class_prediction)
        p_c_given_d = -(category[0])    # Turn it back into a positive int
        print(f'Class prediction: {category[1]}. P(c|doc): {p_c_given_d}')
        return category[1]


    def _predict_category_for_doc(self, document):
        return


    def _get_classes(self, class_filepath):
        # get list containing document index in the first column and class ID in the 2nd column
        # return the set of distinct classes (to be used in training algorithm)
        # we need to keep the document indexes, we do need the total count of documents
        # total count of docs = # of read lines
        classes_filepath = class_filepath
        class_list = self._get_file_by_line(classes_filepath)
        total_num_of_documents = len(class_list)    # should be 800

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
        file = open(filepath)
        content_list = file.readlines()
        file.close()

        for i in range(len(content_list)):
            content_list[i] = content_list[i].rstrip("\n")

        if content_list[len(content_list)-1] == '':
            content_list.pop() # Remove last line 
        
        return content_list

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

    def _get_vocab_list(self, vocab_filepath):
        'Create list for vocabulary in documents'
        filepath = vocab_filepath   #alternatively, filepath = input("Enter the Terms file path: ")
        vocab = self._get_file_by_line(filepath)    
        return vocab

train_class_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_train_classes.txt'
train_matrix_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_train_matrix.txt'
test_class_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_classes.txt'
test_matrix_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_matrix.txt'
terms_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_terms.txt'

classifier = Naive_Bayesian_Classifier(train_class_file, train_matrix_file, test_class_file, test_matrix_file, terms_file)
classifier.naive_bayesian_algorithm()