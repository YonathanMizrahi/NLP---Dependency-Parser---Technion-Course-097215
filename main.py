from data_preprocessing import *
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from LSTM_Model import LSTM_Model
import torch.optim as optim
from utils import *
import matplotlib.pyplot as plt

 

def get_labels(labels_tensor):
    labels = [labels[:sentence_length[i]] for i, labels in enumerate(labels_tensor)]
    return labels

if __name__ == "__main__":

    #Fix manual seed
    torch.manual_seed(0)

    # Hyper-parameters:
    path_train = "data/train.labeled"
    path_test = "data/test.labeled"
    EPOCHS = 1
    WORD_EMBEDDING_DIM = 300
    POS_EMBEDDING_DIM = 25
    HIDDEN_DIM = 125
    BATCH_SIZE = 1 # WARNING: Do not change because it will not work
    LEARNING_RATE = 0.001


    # --- PREPARE THE DATASET ----
    words_dict, pos_dict = get_vocabs([path_train])  # Gets all known vocabularies.

    # Train
    train = PosDataset(words_dict, pos_dict, path_train, padding=True)
    train_data_loader = DataLoader(train.sentences_dataset, batch_size=BATCH_SIZE,  shuffle=False)
    print("Number of Train Tagged Sentences ", len(train))

    # Test
    test = PosDataset(words_dict, pos_dict, path_test, padding=True)
    test_data_loader = DataLoader(test.sentences_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Number of Test Tagged Sentences ", len(test))

    word_to_idx, idx_to_word, _ = train.get_words_embeddings()

    # --- GET SIZE DICT -----------
    word_vocab_size = len(words_dict) # finally not relevant since we use a pre-trained glove
    pos_vocab_size = len(pos_dict)


    # --- MODEL & OPTIMIZER ------
    model = LSTM_Model(BATCH_SIZE, WORD_EMBEDDING_DIM,POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, pos_vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------ CHOOSE DEVICE -------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model.cuda()

    # ------- TRAINING ----------
    print("Training started .......")
    all_train_loss = []
    all_train_UAS = []
    all_test_UAS = []
    all_test_loss = []

    for epoch in range(EPOCHS):
        print('epoch number: ' + str(epoch))
        train_curr_epoch_UAS_accuracy = 0
        test_curr_epoch_UAS_accuracy = 0
        train_curr_epoch_loss = 0
        test_curr_epoch_loss = 0
        train_num_of_words = 0
        test_num_of_words = 0

        for i, input_data in enumerate(train_data_loader):
            if i % 1000 == 0:
                print('\033[32m' + f"{i} Sentences / 5000 Trained " + '\033[0m')

            model.train()
            words_idx_tensor, pos_idx_tensor, train_labels_tensor, sentence_length = input_data
            # Extract only the relevant labels (instead on a tensor with shape [1,250])
            train_labels = get_labels(train_labels_tensor)
            # Feeding our model with the current batch.
            weights = model(words_idx_tensor, pos_idx_tensor, sentence_length)
            # Calculate the negative log likelihood loss from Tutorial 7
            loss = calc_NLLoss(train_labels, weights, sentence_length)
            # Optimize:
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_curr_epoch_loss += loss.item()
            #Calc UAS acc
            train_curr_epoch_UAS_accuracy += calculate_UAS_accuracy(weights, train_labels, sentence_length)
            train_num_of_words += sentence_length.sum() - 1  # We don't count the root as we don't count it in the accuracy.

        # --- PRINT ACCURACY AND LOSS (TRAIN) ----
        train_loss = train_curr_epoch_loss / len(train)
        all_train_loss.append(train_loss)
        train_UAS = train_curr_epoch_UAS_accuracy / train_num_of_words
        train_UAS = train_UAS.item()
        all_train_UAS.append(train_UAS)
        print('\033[1m' + f"Train_Loss : {float(train_loss)} Train_UAC : {float(train_UAS)} " + '\033[0m')

        # --- PRINT ACCURACY AND LOSS (TEST) ----
        for input_data in test_data_loader:
            # Unpack input data
            words_idx_tensor, pos_idx_tensor, test_label_tensor, sentence_length = input_data
            # Extract only the relevant labels (instead on a tensor with shape [1,250])
            test_labels = get_labels(test_label_tensor)
            # Get model predictions for batch
            mat_scores = model(words_idx_tensor, pos_idx_tensor, sentence_length)
            #Compute the loss
            test_loss = calc_NLLoss(test_labels, mat_scores, sentence_length)
            test_curr_epoch_loss += test_loss.item()
            # Update accuracy and number of words counters
            test_curr_epoch_UAS_accuracy += calculate_UAS_accuracy(mat_scores, test_labels, sentence_length)
            # Remove the root to not bias the accuracy
            test_num_of_words += sentence_length.sum() - 1

        test_UAS = (test_curr_epoch_UAS_accuracy / test_num_of_words)
        test_UAS = test_UAS.item()
        test_loss = test_curr_epoch_loss / len(test)
        all_test_UAS.append(test_UAS)
        all_test_loss.append(test_loss)
        print('\033[1m' + f"Test_Loss : {float(test_loss)} Test_UAC : {float(test_UAS)} " + '\033[0m')

    print("Training terminated .......")

    # ------- TESTING ----------
    print("Evaluating Started (over all the test file)")
    model.eval()
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
            test_labels = get_labels(test_label_tensor)
            # Get model predictions for batch
            mat_scores = model(words_idx_tensor, pos_idx_tensor, sentence_length)
            # Update accuracy and number of words counters
            acc += calculate_UAS_accuracy(mat_scores, test_labels,  sentence_length)
            # Remove the root to not bias the accuracy
            num_of_words += sentence_length.sum() - 1
    test_acc = acc / num_of_words
    test_acc = test_acc.item()
    print("Evaluating Ended")
    print('Accuracy over all the test set: ' + str(test_acc))

    print('Saving the model')
    torch.save(model.state_dict(), "lstm_on_VM.pkl")
    print('Model saved.')
    quit()
    # --- Plot the models ----
    # Plot the losses of the train model
    plt.plot(all_train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Convergence graph: train loss as a function of time (epochs)')
    plt.show()

    # Plot the UAS accuracy of the train model
    plt.plot(all_train_UAS)
    plt.xlabel('Epoch')
    plt.ylabel('UAS')
    plt.title('Convergence graph: train UAS acc as a function of time (epochs)')
    plt.show()

    # Plot the losses of the train model
    plt.plot(all_test_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Convergence graph: test loss as a function of time (epochs)')
    plt.show()

    # Plot the UAS accuracy of the test model
    plt.plot(all_test_UAS)
    plt.xlabel('Epoch')
    plt.ylabel('UAS')
    plt.title('Convergence graph: test UAS acc as a function of time (epochs)')
    plt.show()









