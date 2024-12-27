import cv2
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
from PIL import Image
from utils import image_transformation, set_cuda

device = set_cuda()

class Vocabulary:  # Creates vocab for tokens with freq > freq_threshold
    def __init__(self, caption_list, freq_threshold):
        self.caption_list = caption_list
        self.threshold = freq_threshold
        # Adding special tokens
        self.idx2wrd = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unknown>'}  # Maps token ID to the actual token
        self.wrd2idx = {word: idx for idx, word in self.idx2wrd.items()}  # Maps token to its numerical ID
        self.create_vocab()

    def create_vocab(self):
        all_tokens = [word for caption in self.caption_list for word in caption.split()]
        word_counts = Counter(all_tokens)
        index = len(self.idx2wrd)  # Start adding to the dict after the special tokens index
        for word, count in word_counts.items():
            if count >= self.threshold and word not in self.wrd2idx:
                self.wrd2idx[word] = index
                self.idx2wrd[index] = word
                index += 1

    def cap2tensor(self, caption):
        numericalized_caption = [self.wrd2idx['<start>']]  # Adding <start> token ID to the beginning of the caption
        for word in caption.split():
            if word in self.wrd2idx:
                numericalized_caption.append(self.wrd2idx[word])  # If token found in vocab, append its token ID
            else:
                numericalized_caption.append(self.wrd2idx['<unknown>'])  # If not found, add <unknown> ID
        numericalized_caption.append(self.wrd2idx['<end>'])  # Adding <end> token ID at the end of the caption
        return torch.tensor(numericalized_caption)

    def __len__(self):
        return len(self.wrd2idx)


class Image_encoder(nn.Module):
    def __init__(self):
        super(Image_encoder, self).__init__()
        self.resnet = models.resnet101(pretrained=True)  # ResNet101 for feature extraction
        for param in self.resnet.parameters():
            param.requires_grad_(False)
        self.layers_list = list(self.resnet.children())[:-2]  # Remove the last classification layer and its FC
        self.Resnet = nn.Sequential(*self.layers_list)

    def forward(self, image_tensor):
        features = self.Resnet(image_tensor)  # Shape: (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # Shape: (batch_size, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # Flatten to a single tensor
        return features  # Shape: (batch_size, 49, 2048)


class AttentionLayer(nn.Module):  # Soft attention
    def __init__(self, features_dims, hidden_state_dims, attention_dims):
        super().__init__()
        self.attention_dims = attention_dims
        self.U = nn.Linear(features_dims, attention_dims)
        self.W = nn.Linear(hidden_state_dims, attention_dims)
        self.A = nn.Linear(attention_dims, 1)  # Compute attention scores using attention dims for both U & W

    def forward(self, img_features, hidden_state):
        u_hs = self.U(img_features)  # Shape: (batch_size, 49, attention_dims)
        w_hs = self.W(hidden_state)  # Shape: (batch_size, attention_dims)
        combined_states = torch.tanh(u_hs + w_hs.unsqueeze(1))  # Outputs: (batch_size, 49, attention_dims)
        attention_scores = self.A(combined_states)  # Shape: (batch_size, 49, 1)
        attention_scores = attention_scores.squeeze(2)  # Shape: (batch_size, 49)
        alpha = F.softmax(attention_scores, dim=1)  # Attention weights
        context_vector = img_features * alpha.unsqueeze(2)  # Shape: (batch_size, 49, features_dims)
        context_vector = context_vector.sum(dim=1)  # Weighted image features vector
        return alpha, context_vector  # Alpha for visualization, context_vector for RNN feeding


class Attention_Based_Decoder(nn.Module):
    def __init__(self, features_dims, hidden_state_dims, attention_dims, word_emb_dims, vocab_size, drop_prob):
        super().__init__()
        self.vocab_size = vocab_size
        self.dropout_layer = nn.Dropout(drop_prob)
        self.attention_layer = AttentionLayer(features_dims, hidden_state_dims, attention_dims)
        self.tokens_embedding = nn.Embedding(vocab_size, word_emb_dims)
        self.hidden_state_init = nn.Linear(features_dims, hidden_state_dims)  # FC to initialize hidden_state
        self.cell_state_init = nn.Linear(features_dims, hidden_state_dims)  # FC to initialize cell_state
        self.lstm_cell = nn.LSTMCell(word_emb_dims + features_dims, hidden_state_dims, bias=True)
        self.fcl = nn.Linear(hidden_state_dims, vocab_size)

    def init_hidden_state(self, image_features_tensor):
        features_mean = image_features_tensor.mean(dim=1)
        h = self.hidden_state_init(features_mean)
        c = self.cell_state_init(features_mean)
        return h, c

    def forward(self, batch_images_features, batch_captions_tensors):
        captions_len = len(batch_captions_tensors[0]) - 1  # Max caption length
        batch_size = batch_captions_tensors.size(0)
        features_size = batch_images_features.size(1)  # Size of feature vector
        embedded_tokens = self.tokens_embedding(batch_captions_tensors)  # Map each token ID to its embedding
        hidden_state, cell_state = self.init_hidden_state(batch_images_features)
        preds = torch.zeros(batch_size, captions_len, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, captions_len, features_size).to(device)

        for wrd_index in range(captions_len):
            alpha, context_vector = self.attention_layer(batch_images_features, hidden_state)
            current_token_emb = embedded_tokens[:, wrd_index]  # Teacher forcing
            lstm_input = torch.cat((current_token_emb, context_vector), dim=1)
            hidden_state, cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))
            tokens_probs = self.fcl(self.dropout_layer(hidden_state))
            preds[:, wrd_index] = tokens_probs  # Next token probabilities
            alphas[:, wrd_index] = alpha  # Alpha for current token
        return preds, alphas


class Encoder_Decoder_Model(nn.Module):
    def __init__(self, features_dims, hidden_state_dims, attention_dims, word_emb_dims, vocab_size, drop_prob):
        super().__init__()
        self.img_encoder = Image_encoder()
        self.decoder = Attention_Based_Decoder(features_dims, hidden_state_dims, attention_dims, word_emb_dims, vocab_size, drop_prob)

    def forward(self, batch_images, batch_tokenized_captions):
        image_features = self.img_encoder(batch_images)
        probs, alphas = self.decoder(image_features, batch_tokenized_captions)
        return probs, alphas

    def predict(self, image, vocab, max_cap_len=20, debugging=False):
        self.eval()
        with torch.no_grad():
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = image_transformation(img).unsqueeze(0).to(device)
            image_features = self.img_encoder(img)
            if debugging:
                print(f"predict input image_Shape:{image_features.shape}")
            hidden_state, cell_state = self.decoder.init_hidden_state(image_features)
            caption = [vocab.idx2wrd[1]]
            token = torch.tensor(vocab.wrd2idx["<start>"]).unsqueeze(0).to(device)
            attentions = []

            for i in range(max_cap_len):
                alpha, context_vector = self.decoder.attention_layer(image_features, hidden_state)
                if debugging:
                    print(i, "-attention map for token:", vocab.idx2wrd[token.item()], "is", alpha.shape)
                attentions.append(alpha.cpu().detach().numpy())
                current_token_emb = self.decoder.tokens_embedding(token)
                lstm_input = torch.cat((current_token_emb.squeeze(1), context_vector), dim=1)
                hidden_state, cell_state = self.decoder.lstm_cell(lstm_input, (hidden_state, cell_state))
                tokens_prob = self.decoder.fcl(hidden_state)
                next_token = tokens_prob.argmax(dim=1).item()
                next_word = vocab.idx2wrd[next_token]
                caption.append(next_word)
                if next_word == "<end>":
                    break
                token = torch.tensor([next_token]).unsqueeze(0).to(device)

            if debugging:
                print("attention shape:", np.array(attentions).shape)
                print("caption_length:", len(caption))
            return attentions, caption
