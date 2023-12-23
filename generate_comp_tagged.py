from LSTM_Model import *
from data_preprocessing import *
from utils import *


if __name__ == "__main__":
    # Fix manual seed
    torch.manual_seed(0)

    # Paths
    path_train = "data/train.labeled"
    path_test = "data/test.labeled"
    path_comp = "data/comp.unlabeled"
    path_comp_labeled = 'comp_209948728_931188684.labeled'

    # Hyper-parameters:
    WORD_EMBEDDING_DIM = 300
    POS_EMBEDDING_DIM = 25
    HIDDEN_DIM = 125
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001

    # --- PREPARE THE DATASET ----
    print("Preparing the dataset")
    words_dict, pos_dict = get_vocabs([path_train])  # Gets all known vocabularies.
    comp = PosDataset(words_dict, pos_dict, path_comp, padding=True, competition=True)
    comp_data_loader = DataLoader(comp.sentences_dataset, batch_size=1,  shuffle=False)
    word_to_idx, idx_to_word, _ = comp.get_words_embeddings()

    # --- GET SIZE DICT -----------
    word_vocab_size = len(words_dict)
    pos_vocab_size = len(pos_dict)


    # --- Load the pre-trained LSTM model -----
    print("Loading the pre-trained LSTM model")
    
    PATH = 'lstm_on_GPU_30epochs.pkl'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = LSTM_Model(BATCH_SIZE, WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM,
                       word_vocab_size, pos_vocab_size)
    model.load_state_dict(torch.load(PATH, map_location=device))
    #model.load_state_dict(torch.load(PATH))

    # ------ CHOOSE DEVICE -------

    if use_cuda:
        model.cuda()
    model = model.eval()

    """
    First, we will check that our model was loaded correctly. 
    To do it, we will load the test.labeled file and check that our accuracy is still higher than 80%
    The exact accuracy should be: 
    """
    with torch.no_grad():
        print("Checking that the model was correctly loaded (by checking the acc on the test.labeled file)")
        print("Evaluating Started (over all the test file)")
        test = PosDataset(words_dict, pos_dict, path_test, padding=True)
        test_data_loader = DataLoader(test.sentences_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("Number of Test Tagged Sentences ", len(test))

        # Initialize variables
        num_of_sentences = len(test)
        acc = 0
        num_of_words = 0

        # Disable gradient calculation
        with torch.no_grad():
            for input_data in test_data_loader:
                # Unpack input data
                words_idx_tensor, pos_idx_tensor, test_label_tensor, sentence_length = input_data
                # Extract only the relevant labels (instead on a tensor with shape [1,250])
                test_labels = [labels[:sentence_length[i]] for i, labels in enumerate(test_label_tensor)]
                # Get model predictions
                mat_scores = model(words_idx_tensor, pos_idx_tensor, sentence_length)
                # Update accuracy and number of words counters
                acc += calculate_UAS_accuracy(mat_scores, test_labels, sentence_length)
                #Remove the root to not bias the accuracy
                num_of_words += sentence_length.sum() - 1
        test_acc = acc / num_of_words
        test_acc = test_acc.item()
        print("Evaluating Ended")
        print('Accuracy over all the test set: ' + str(test_acc))
        if test_acc < 0.8:
            print("It seems that we had a problem with our loaded model... The generated comp will be problematic...")
        else:
            print("It seems that the pre-trained model was correctly loaded.")
            print("Let's now create the comp.labeled file")

    """ Now we will create the comp.labeled file using our pre-trained model """
    with torch.no_grad():
        print("Evaluating Started (over all the comp file)")
        print("Number of Comp Tagged Sentences ", len(comp))
        trees = []
        for input_data in comp_data_loader:
            # Unpack input data
            words_idx_tensor, pos_idx_tensor, _, sentence_length = input_data
            # Get model predictions
            mat_scores = model(words_idx_tensor, pos_idx_tensor, sentence_length)
            # Get edge scores for current sentence
            trees.append(decode_mst(np.array(mat_scores[:, 0].detach().cpu()).reshape((sentence_length, sentence_length))
                                    [:sentence_length[0], :sentence_length[0]], sentence_length[0], has_labels=False)[0])

        current_tree_idx = 0
        current_word_idx = 1

        with open(path_comp, 'r') as f_unlabeled:
            with open(path_comp_labeled, 'w') as f_labeled:

                for i, line in enumerate(f_unlabeled):
                    split_line = line.split('\t')

                    if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                        current_word_idx = 1
                        current_tree_idx += 1
                        f_labeled.writelines(''.join(split_line))
                        continue

                    split_line[6] = str(trees[current_tree_idx][current_word_idx])
                    current_word_idx += 1

                    f_labeled.writelines('\t'.join(split_line))
    print("Evaluate end")
    print("File comp_209948728_931188684.labeled is ready.")
