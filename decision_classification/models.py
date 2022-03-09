# Importing relevant libraries and dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import DistilBertForSequenceClassification, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput

class SkeletalDistilBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distilbert = DistilBertForSequenceClassification(config=config)
    
    def forward(self, input_ids, labels):
        return self.distilbert(input_ids=input_ids, labels=labels)

class DistilBertWithExaminerID(nn.Module):
    def __init__(self, config, hidden_dim: int = 768, mlp_dim: int = 128, num_embeddings: int = 10, extras_dim: int = 128, dropout: float = 0.1,
                 ex_id_map: dict = None):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.ex_id_map = ex_id_map

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_embeddings, extras_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),            
            nn.Linear(mlp_dim, self.num_labels)
        )
        # self.classifier = nn.Linear(config.dim, config.num_labels)
        
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        examiner_id=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embedding = self.embedding(torch.tensor([self.ex_id_map[_id.item()] for _id in examiner_id])) # each _id is a 0-dim tensor we need to unpack in order to index
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dropout(pooled_output)
        concat_output = torch.cat((pooled_output, embedding), dim=1) # (bs, dim + extras_dim)
        # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        # pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        # pooled_output = self.dropout(pooled_output)  # (bs, dim)
        # logits = self.classifier(pooled_output)  # (bs, num_labels)
        logits = self.mlp(concat_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

class LogisticRegression (nn.Module):
    """ Simple logistic regression model """

    def __init__ (self, vocab_size, embed_dim, n_classes, pad_idx):
        super (LogisticRegression, self).__init__ ()
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Linear layer
        self.fc = nn.Linear (embed_dim, n_classes)
        
    def forward (self, input_ids):
        # Apply the embedding layer
        embed = self.embedding(input_ids)
        # Apply the linear layer
        output = self.fc (embed)
        # Take the sum of the overeall word embeddings for each sentence
        output = output.sum (dim=1)
        return output


class BasicCNNModel (nn.Module):
    """ Simple 2D-CNN model """
    def __init__(self, vocab_size, embed_dim, n_classes, n_filters, filter_sizes, dropout, pad_idx):
        super(BasicCNNModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Conv layer
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes])
        # Linear layer
        self.fc = nn.Linear(
            len(filter_sizes) * n_filters, 
            n_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        # embed = [batch size, sent len, emb dim]
        embed = embed.unsqueeze(1)
        # embed = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]    
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        output = self.fc(cat) #.sigmoid ().squeeze()
        return output


class BigCNNModel (nn.Module):
    """ Slightly more sophisticated 2D-CNN model """
    def __init__(self, vocab_size, embed_dim, pad_idx, n_classes, n_filters=25, filter_sizes=[3,4,5], dropout=0.25):
        super(BigCNNModel, self).__init__()
        print(f'filter_sizes: {filter_sizes}')
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Conv layers
        self.convs_v1 = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes[0]])
        self.convs_v2 = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes[1]])
        self.convs_v3 = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes[2]])
        # Linear layer
        self.fc = nn.Linear(
            len(filter_sizes) * n_filters * 3, 
            n_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        # embed = [batch size, sent len, emb dim]
        embed = embed.unsqueeze(1)
        # embed = [batch size, 1, sent len, emb dim]
        conved_v1 = [F.relu(conv(embed)).squeeze(3) for conv in self.convs_v1]
        conved_v2 = [F.relu(conv(embed)).squeeze(3) for conv in self.convs_v2]
        conved_v3 = [F.relu(conv(embed)).squeeze(3) for conv in self.convs_v3]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_v1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_v1]
        pooled_v2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_v2]
        pooled_v3 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_v3]
        # pooled_n = [batch size, n_filters]
        cat_v1 = self.dropout(torch.cat(pooled_v1, dim = 1))
        cat_v2 = self.dropout(torch.cat(pooled_v2, dim = 1))
        cat_v3 = self.dropout(torch.cat(pooled_v3, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        cat = torch.cat([cat_v1, cat_v2, cat_v3], dim=1)
        output = self.fc(cat) #.sigmoid ().squeeze()
        return output