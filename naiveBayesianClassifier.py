# Crystal A. Contreras  Spring CSC-480  Assignment 4

def naive_bayesian_algorithm():
    """
        Split into 2 parts: training component & testing component.
            Training component returns a set of all priors for each category: P(c)
            and set of conditional probabilities for each term given category: P(term|c)
    """
    ##### TRAINING COMPONENT #####
    # Get data representation vector
    data_vector = _get_data_vector(class_file, matrix_file, terms_file)
    print('Data vecor: ', data_vector)

    # Assign data objects 
    # Create list for vocabulary in documents
    vocab = data_vector['vocabulary']
    print('Vocab: ', vocab)

    # Get class labels
    categories = data_vector['classes']
    print('Categories: ', categories)

    total_num_of_documents = _get_doc_count(categories)
    print('Total documents: ', total_num_of_documents)
        
    # for label in categories.keys():
    #     p_c = categories[label] / total_num_of_documents
    #     print(f'Probability of class {label}: {p_c}')

def _get_classes(class_filepath):
    # get list containing document index in the first column and class ID in the 2nd column
    # return the set of distinct classes (to be used in training algorithm)
    # we need to keep the document indexes, we do need the total count of documents
    # total count of docs = # of read lines
    classes_filepath = class_filepath
    class_list = _get_file_by_line(classes_filepath)
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

def _get_data_vector(class_file, matrix_file, terms_file):
    """ 
        Object that returns class IDs (aka categories), 
        & documents with counts for each word 
    """
    classes = _get_classes(class_file)
    vocabulary = _get_vocab_list(terms_file)
    train_matrix = _get_file_by_line(matrix_file)
    docs_with_vocab_matrix = _get_documents_with_vocab(vocabulary, train_matrix)

    for i in classes:
        for key in classes[i].keys():
            classes[i][key] = docs_with_vocab_matrix[key]

    data_vector = {
        'classes': classes,
        'vocabulary': vocabulary
    }

    return data_vector

def _get_doc_count(classes):
    total_docs = 0

    for c in range(len(classes)):
        total_docs += len(classes[str(c)])

    return total_docs

def _get_documents_with_vocab(sample_vocab, sample_matrix):
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

def _get_file_by_line(filepath):
    file = open(filepath)
    content_list = file.readlines()
    file.close()

    for i in range(len(content_list)):
        content_list[i] = content_list[i].rstrip("\n")

    if content_list[len(content_list)-1] == '':
        content_list.pop() # Remove last line 
    
    return content_list

def _get_vocab_list(vocab_filepath):
    'Create list for vocabulary in documents'
    filepath = vocab_filepath   #alternatively, filepath = input("Enter the Terms file path: ")
    vocab = _get_file_by_line(filepath)    
    return vocab


class_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_classes.txt'
matrix_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_matrix.txt'
terms_file = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/newsgroups/mini_terms.txt'

naive_bayesian_algorithm()