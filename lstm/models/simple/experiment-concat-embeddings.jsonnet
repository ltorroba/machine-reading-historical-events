local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local getTokenIndexer(type) =
  if type == "glove-fasttext" then {
          "tokens_glove": {
              "type": "single_id",
              "lowercase_tokens": true
          },
          "tokens_fasttext": {
              "type": "single_id",
              "lowercase_tokens": false
          }
      }
  else if type == "bert-base" then {
          "tokens_bert": {
              "type": "bert-pretrained",
              "pretrained_model": "bert-base-uncased",
              "do_lowercase": true
          }
      }
  else if type == "bert-large" then {
          "tokens_bert": {
              "type": "bert-pretrained",
              "pretrained_model": "bert-large-uncased",
              "do_lowercase": true
          }
      }
  else error "unknown embedding type: " + type;

local getTokenEmbedder(type, embeddings_root) =
  if type == "glove-fasttext" then {
        "tokens_glove": {
            "type": "embedding",
            "embedding_dim": 200,
            "pretrained_file": embeddings_root + "/glove/glove.6B.200d.txt",
            "trainable": false
        },
        "tokens_fasttext": {
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": embeddings_root + "/fasttext/wiki-news-300d-1M-subword.vec",
            "trainable": false
        }
    }
  else if type == "bert-base" then {
        "tokens_bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": false
        }
    }
  else if type == "bert-large" then {
        "tokens_bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-large-uncased",
            "requires_grad": false
        }
    }
  else error "unknown embedding type: " + type;

local getTokenEmbedderAllowUnmatchedKeys(type) =
  if type == "glove-fasttext" then false
  else if type == "bert-base" then true
  else if type == "bert-large" then true
  else error "unknown embedding type: " + type;

local getEmbeddingDim(type) =
  if type == "glove-fasttext" then 200 + 300  # glove_dim + fasttext_dim
  else if type == "bert-base" then 768
  else if type == "bert-large" then 1024
  else error "unknown embedding type: " + type;

local getEmbeddingEncoder(embedding_type, encoder_hidden_dim, encoder_dropout, encoder_num_layers) =
  if embedding_type == "glove-fasttext" then {
        "type": "lstm",
        "input_size": getEmbeddingDim(embedding_type),
        "hidden_size": encoder_hidden_dim,
        "dropout": encoder_dropout,
        "num_layers": encoder_num_layers,
        "bidirectional": true
    }
  else if embedding_type == "bert-base" then {
        "type": "bert_pooler",
        "pretrained_model": "bert-base-uncased",
        "dropout": encoder_dropout
    }
  else if embedding_type == "bert-large" then {
        "type": "bert_pooler",
        "pretrained_model": "bert-large-uncased",
        "dropout": encoder_dropout
    }
  else error "unknown embedding type: " + embedding_type;

local getEmbeddingEncoderOutputDim(embedding_type, hidden_dim) =
  if embedding_type == "glove-fasttext" then std.parseInt(hidden_dim) * 2
  else if embedding_type == "bert-base" then getEmbeddingDim(embedding_type)
  else if embedding_type == "bert-large" then getEmbeddingDim(embedding_type)
  else error "unknown embedding type: " + embedding_type;

local getCIAttention(type, encoder_out_dim) =
  if type == "uniform-attention" then {"type": "uniform-attention"}
  else if type == "dot_product" then {"type": "dot_product"}
  else if type == "linear-concat" then
    #{"type": "linear", "combination": "x,y", "tensor_1_dim": encoder_out_dim, "tensor_2_dim": encoder_out_dim}
    error "NOTE: AllenNLP implementation of this form of attention crashes on this mode when a batch's max_ci_sentences_in_inp == 1"
  else if type == "additive" then
    {"type": "additive", "vector_dim": encoder_out_dim, "matrix_dim": encoder_out_dim}
  else error "invalid attention type: " + type;

local getDatasetMin(dataset) =
  if dataset == "wotd" then 1302
  else if dataset == "otd" then 1
  else if dataset == "otd2" then 1
  else error "unknown dataset type: " + dataset;

local getDatasetMax(dataset) =
  if dataset == "wotd" then 2018
  else if dataset == "otd" then 2018
  else if dataset == "otd2" then 2020
  else error "unknown dataset type: " + dataset;

local getDatasetMean(dataset) =
  if dataset == "wotd" then 1819.0
  else if dataset == "otd" then 1913.0
  else if dataset == "otd2" then 1908.0
  else error "unknown dataset type: " + dataset;

local getDatasetStd(dataset) =
  if dataset == "wotd" then 156.0
  else if dataset == "otd" then 170.0
  else if dataset == "otd2" then 177.0
  else error "unknown dataset type: " + dataset;

local getDatasetFilename(dataset) =
  if dataset == "wotd" then "wotd.pkl"
  else if dataset == "otd" then "otd.pkl"
  else if dataset == "otd2" then "otd2.pkl"
  else error "unknown dataset type: " + dataset;

local getDatasetSize(dataset, debug_mode, hyperparam_mode) =
  if debug_mode then null               # Testing: If we are testing we don't care about dataset size
  else if hyperparam_mode then
    if dataset == "wotd" then 1089      # Tuning: 20% for WOTD
    else if dataset =="otd" then 6003   # Tuning: 10% for OTD
    else if dataset =="otd2" then 5718   # Tuning: 10% for OTD2
    else error "unknown dataset type: " + dataset
  else null;                            # Normal: run on whole dataset

local checkModesValid(debug_mode, hyperparam_mode) =
  if debug_mode && hyperparam_mode then error "cannot be both hyperparam & debug mode"
  else null;

local use_ci = stringToBool(std.extVar('use_ci'));

local debug_mode = stringToBool(std.extVar('debug_mode'));
local hyperparam_mode = stringToBool(std.extVar('hyperparam_mode'));
local batch_check_mode = stringToBool(std.extVar('batch_check_mode'));

local predictor_type = std.extVar('predictor_type'); # "cls" or "reg"

local _ = checkModesValid(debug_mode, hyperparam_mode);

local attention_type = std.extVar('attention_type');

local embedding_type = std.extVar('embedding_type');

local encoder_hidden_dim = std.extVar('encoder_hidden_dim');
local encoder_num_layers = std.extVar('encoder_num_layers');
local encoder_dropout = std.extVar('encoder_dropout');
local encoder_output_dim = encoder_hidden_dim * 2;

local predictor_hidden_dim = std.extVar('predictor_hidden_dim');
local predictor_num_layers = std.extVar('predictor_num_layers');
local predictor_dropout = std.extVar('predictor_dropout');

local num_epochs = std.extVar('num_epochs');
local batch_size = std.extVar('batch_size');
local learning_rate = std.extVar('learning_rate');

local grad_clipping = 10.0;
local cuda_device = if debug_mode then -1 else 0;

local dataset = std.extVar('dataset'); # 'wotd', 'otd' or 'otd2'
local normalize_mean = if predictor_type == "reg" then getDatasetMean(dataset) else 0.0;  # If we are regressing, then we
local normalize_std = if predictor_type == "reg" then getDatasetStd(dataset) else 1.0;    # would like to normalize the outputs.

local dataset_root = "../data";
local dataset_file = getDatasetFilename(dataset);
local dataset_limit_number = getDatasetSize(dataset, debug_mode, hyperparam_mode);

local embeddings_root = "embeddings";

{
    "train_data_path": dataset_root + "/train/" + dataset_file,
    "validation_data_path": dataset_root + "/validation/" + dataset_file,
    "dataset_reader": {
        "type": "heo-reader",
        "token_indexers": getTokenIndexer(embedding_type),
        "limit_number": dataset_limit_number,
        "normalize_outputs": [normalize_mean, normalize_std]
    },
    "model": {
        "type": "simple-regressor",
        "predictor_type": predictor_type,
        "dropout": predictor_dropout,
        "predictor_num_layers": predictor_num_layers,
        "predictor_hidden_features": predictor_hidden_dim,
        "use_ci": use_ci,
        "ci_attention": getCIAttention(attention_type, getEmbeddingEncoderOutputDim(embedding_type, encoder_hidden_dim)),
        "word_embeddings": {
            "type": "basic",
            "token_embedders": getTokenEmbedder(embedding_type, embeddings_root),
            "allow_unmatched_keys": getTokenEmbedderAllowUnmatchedKeys(embedding_type)
        },
        "encoder": getEmbeddingEncoder(embedding_type, encoder_hidden_dim, encoder_dropout, encoder_num_layers),
        "normalize_outputs": [normalize_mean, normalize_std],
        "year_range": [getDatasetMin(dataset), getDatasetMax(dataset)]
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["information", "list_num_tokens"], ["event", "num_tokens"]],
        "cache_instances": true,
        "biggest_batch_first": batch_check_mode
    },
    "trainer": {
        "num_epochs": num_epochs,
        "patience": 25,
        "optimizer": {
            "type": "adam",
            "lr": learning_rate
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau"
        },
        "grad_norm": grad_clipping,
        "cuda_device": cuda_device,
        "should_log_learning_rate": true
    }
}
