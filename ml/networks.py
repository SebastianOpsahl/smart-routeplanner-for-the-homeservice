import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

class PointerNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=128):
        super(PointerNetwork, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTMCell(embedding_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        # Additional layer for before/after decision
        self.before_after_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, last_hidden=None):
        embedded = F.relu(self.embedding(x))
        encoder_outputs, (hidden, cell) = self.encoder(embedded)

        if last_hidden is None:
            hidden, cell = self.init_hidden(x)

        hidden, cell = self.decoder(embedded[:, -1, :], (hidden, cell))

        # Attention mechanism to determine visits
        attention_weights = F.softmax(self.attention(hidden.unsqueeze(1) + encoder_outputs), dim=1)
        visit_logits = attention_weights.squeeze(-1)

        # Decide on two visits based on attention logits
        visit_probs = F.softmax(visit_logits, dim=1)
        dist = Categorical(visit_probs)
        visit1_index = dist.sample()
        log_prob1 = dist.log_prob(visit1_index)

        # Mask the selected index to avoid re-selecting the same visit
        mask = torch.ones_like(visit_probs)
        mask.scatter_(1, visit1_index.unsqueeze(0), 0)
        visit_probs = visit_probs * mask
        visit_probs = visit_probs / visit_probs.sum(dim=1, keepdim=True)
        dist = Categorical(visit_probs)
        visit2_index = dist.sample()
        log_prob2 = dist.log_prob(visit2_index)

        # Classifier for deciding before/after
        before_after_logit = self.before_after_classifier(hidden)
        before_after_prob = torch.sigmoid(before_after_logit).squeeze()
        dist = Bernoulli(before_after_prob)
        before_after = dist.sample()   
        log_prob_ba = dist.log_prob(before_after)
        before_after = before_after.long()

        return visit1_index, visit2_index, before_after, log_prob1, log_prob2, log_prob_ba

    def init_hidden(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size, self.encoder.hidden_size).to(x.device)
        cell = torch.zeros(batch_size, self.encoder.hidden_size).to(x.device)
        return hidden, cell

class Critic(nn.Module):
    '''Setting dimensions of the network and initializes it'''
    def __init__(self, combined_feature_size, hidden_dim=128, output_dim=1):
        super(Critic, self).__init__()
        self.feature_encoder = nn.LSTM(combined_feature_size, hidden_dim, batch_first=True)
        
        # fully connected layers to process the combined feature encoding
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, combined_features):
        _, (hidden, _) = self.feature_encoder(combined_features)
        
        x = F.relu(self.fc1(hidden[-1]))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value