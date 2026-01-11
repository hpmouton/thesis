# Julia Model Implementation Prompts
## IDP Recommender System - Deep Learning Models

Use these prompts with an AI assistant or as implementation guides for building your thesis models in Julia.

---

## 🔧 Environment Setup Prompt

```
I'm building an Individual Development Plan (IDP) Recommender System for my thesis using Julia. 

Please help me set up a Julia project with the following packages:
- Flux.jl for deep learning
- MLDatasets for data handling
- DataFrames.jl for data manipulation
- CSV.jl for loading data
- Plots.jl and StatsPlots.jl for visualization
- BSON.jl for model saving
- GraphNeuralNetworks.jl for GNN implementation
- Transformers.jl for transformer models
- CUDA.jl for GPU acceleration (optional)

Create a Project.toml with appropriate dependencies and provide the initial setup code including:
1. Package installation commands
2. Basic project structure
3. GPU detection and setup
4. Random seed setting for reproducibility
```

---

## 📊 Data Preprocessing Prompt

```
I have employee data for an IDP Recommender System with the following features:
- Employee ID
- Skills (list of skill names)
- Performance reviews (text)
- Career goals (text)
- Training history (courses completed)
- Job role
- Years of experience
- Manager ratings

Please write Julia code to:
1. Load and clean this data using DataFrames.jl
2. Handle missing values appropriately
3. Encode categorical variables (skills, job roles)
4. Create user-item interaction matrix for collaborative filtering
5. Tokenize and embed text data (career goals, performance reviews)
6. Split data into train/validation/test sets (70/15/15)
7. Create DataLoader objects for batch training

Include functions for:
- normalize_features(df::DataFrame)
- create_interaction_matrix(df::DataFrame)
- encode_skills(skills::Vector{String})
- prepare_batches(data, batch_size::Int)
```

---

## 🧠 Model 1: Neural Collaborative Filtering (NCF) Prompt

```
Implement a Neural Collaborative Filtering (NCF) model in Julia using Flux.jl for an IDP Recommender System.

The model should predict which development activities (training courses, skills to develop) to recommend to employees based on:
- Employee embeddings (learned from employee features)
- Item embeddings (learned from development activity features)
- Interaction history between employees and activities

Architecture requirements:
1. Embedding layers for users and items (embedding_dim = 64)
2. GMF (Generalized Matrix Factorization) component
3. MLP (Multi-Layer Perceptron) component with layers [128, 64, 32]
4. NeuMF fusion layer combining GMF and MLP
5. Output layer with sigmoid activation for rating prediction

Please provide:
1. Model struct definition using Flux
2. Forward pass implementation
3. Loss function (Binary Cross-Entropy + regularization)
4. Training loop with:
   - Adam optimizer (lr=0.001)
   - Early stopping
   - Learning rate scheduling
5. Evaluation metrics: Precision@K, Recall@K, NDCG@K, Hit Rate
6. Model saving/loading functions

Include code for:
- struct NCF
- function (model::NCF)(user_ids, item_ids)
- function train_ncf!(model, train_loader, val_loader; epochs=100)
- function evaluate_ncf(model, test_loader)
```

---

## 🔄 Model 2: RNN/LSTM for Sequential Learning Prompt

```
Implement an LSTM-based model in Julia using Flux.jl to capture sequential patterns in employee learning behavior for the IDP Recommender System.

The model should predict the next recommended development activity based on an employee's training history sequence.

Input features:
- Sequence of completed training courses (embedded)
- Time gaps between courses
- Performance improvement after each course

Architecture requirements:
1. Embedding layer for training courses (embedding_dim = 64)
2. 2-layer Bidirectional LSTM (hidden_size = 128)
3. Attention mechanism over LSTM outputs
4. Fully connected layers [256, 128]
5. Output layer predicting next course (softmax over all courses)

Please provide:
1. Sequence padding and batching utilities
2. Model struct with LSTM layers
3. Attention mechanism implementation
4. Training loop with teacher forcing
5. Beam search for inference
6. Evaluation: Accuracy, MRR (Mean Reciprocal Rank)

Include code for:
- struct SequentialRecommender
- function pad_sequences(sequences, max_len)
- function attention(query, keys, values)
- function (model::SequentialRecommender)(sequence)
- function train_lstm!(model, data; epochs=50)
- function predict_next_k(model, history, k=5)
```

---

## 🕸️ Model 3: Graph Neural Network (GNN) Prompt

```
Implement a Graph Neural Network in Julia using GraphNeuralNetworks.jl for the IDP Recommender System.

The model should leverage relationships between:
- Employees (connected if same department/role)
- Skills (connected if commonly learned together)
- Training courses (connected if prerequisite relationships)
- Career paths (connected if lead to similar roles)

Graph structure:
- Heterogeneous graph with multiple node types
- Edge types: employee-has-skill, course-teaches-skill, skill-prerequisite-skill
- Node features: embeddings for each entity type

Architecture requirements:
1. Heterogeneous graph construction
2. GraphSAGE or GAT layers (3 layers, hidden_dim = 128)
3. Message passing for 3 hops
4. Readout layer for node-level predictions
5. Link prediction head for employee-course recommendations

Please provide:
1. Graph construction from employee/skill/course data
2. GNN model definition
3. Neighbor sampling for scalability
4. Training with negative sampling
5. Evaluation: AUC-ROC, Precision@K

Include code for:
- function build_knowledge_graph(employees, skills, courses)
- struct IDPGraphModel
- function message_passing(g, x)
- function (model::IDPGraphModel)(g)
- function train_gnn!(model, graph; epochs=100)
- function recommend_courses(model, graph, employee_id, k=5)
```

---

## 🤖 Model 4: Transformer for Text Understanding Prompt

```
Implement a Transformer-based model in Julia for understanding employee career goals and generating personalized IDP recommendations.

The model should:
1. Encode employee career goal text descriptions
2. Encode job role descriptions and skill requirements
3. Match employees to relevant development opportunities
4. Generate explanations for recommendations

Architecture requirements:
1. Tokenizer for text processing
2. Positional encoding
3. Multi-head self-attention (8 heads, d_model=256)
4. Transformer encoder (6 layers)
5. Cross-attention between employee goals and course descriptions
6. Classification head for recommendation scoring

Please provide:
1. Text tokenization and vocabulary building
2. Positional encoding implementation
3. Multi-head attention mechanism
4. Transformer encoder block
5. Full model for text matching
6. Training with contrastive loss
7. Inference for ranking courses

Include code for:
- struct TransformerEncoder
- function positional_encoding(seq_len, d_model)
- function multi_head_attention(Q, K, V, num_heads)
- function encode_text(model, text)
- function compute_similarity(goal_embedding, course_embeddings)
- function train_transformer!(model, data; epochs=50)
```

---

## 🔀 Model 5: Hybrid Ensemble Model Prompt

```
Create a hybrid ensemble model in Julia that combines NCF, LSTM, GNN, and Transformer models for the IDP Recommender System.

The ensemble should:
1. Take outputs from all four models
2. Learn optimal weights for each model's predictions
3. Handle cases where some models may not have predictions
4. Provide confidence scores for recommendations

Architecture:
1. Feature extraction from each model's output
2. Attention-based fusion mechanism
3. Meta-learner (small neural network) for final prediction
4. Uncertainty estimation using dropout at inference

Please provide:
1. Model loading utilities for all base models
2. Ensemble architecture definition
3. Training the meta-learner while freezing base models
4. Weighted voting mechanism
5. Final recommendation ranking
6. Explainability: which model contributed most

Include code for:
- struct HybridEnsemble
- function load_base_models(model_paths)
- function fuse_predictions(ncf_pred, lstm_pred, gnn_pred, transformer_pred)
- function (model::HybridEnsemble)(employee_data)
- function train_ensemble!(model, data; epochs=20)
- function explain_recommendation(model, employee_id, course_id)
```

---

## 📈 Evaluation and Comparison Prompt

```
Create a comprehensive evaluation framework in Julia to compare all models for the IDP Recommender System thesis.

Metrics to implement:
1. Precision@K (K = 1, 5, 10)
2. Recall@K
3. F1-Score@K
4. NDCG@K (Normalized Discounted Cumulative Gain)
5. MAP (Mean Average Precision)
6. MRR (Mean Reciprocal Rank)
7. AUC-ROC for binary relevance
8. Coverage (% of items recommended)
9. Diversity (intra-list similarity)
10. Novelty (popularity-weighted)

Please provide:
1. Metric computation functions
2. Cross-validation framework (5-fold)
3. Statistical significance tests (paired t-test, Wilcoxon)
4. Visualization functions for:
   - Precision-Recall curves
   - ROC curves
   - Performance bar charts comparing models
   - Training loss curves
5. Results table generation (LaTeX format)

Include code for:
- function precision_at_k(predictions, ground_truth, k)
- function ndcg_at_k(predictions, ground_truth, k)
- function cross_validate(model, data; k_folds=5)
- function statistical_test(results_a, results_b)
- function generate_latex_table(results_dict)
- function plot_comparison(results_dict)
```

---

## 💾 Model Persistence and Deployment Prompt

```
Create utilities for saving, loading, and deploying the IDP Recommender models in Julia.

Requirements:
1. Save model weights using BSON.jl
2. Save training history and hyperparameters
3. Version control for model artifacts
4. Load models for inference
5. Create inference API endpoint (using Genie.jl or HTTP.jl)
6. Batch prediction capabilities
7. Model performance monitoring

Please provide:
1. Model serialization functions
2. Checkpoint saving during training
3. Best model selection based on validation metrics
4. Simple REST API for predictions
5. Logging and monitoring utilities

Include code for:
- function save_model(model, path; metadata=Dict())
- function load_model(path)
- function save_checkpoint(model, epoch, metrics, path)
- function setup_api_endpoint(model, port=8080)
- function predict_batch(model, employee_ids)
- function log_prediction(employee_id, recommendations, confidence)
```

---

## 🚀 Quick Start Example

```julia
# Example usage after implementing all components

using Flux, CUDA, DataFrames, CSV

# 1. Load and preprocess data
df = CSV.read("employee_data.csv", DataFrame)
train_data, val_data, test_data = preprocess_and_split(df)

# 2. Initialize models
ncf_model = NCF(num_users, num_items, embedding_dim=64)
lstm_model = SequentialRecommender(vocab_size, hidden_size=128)
gnn_model = IDPGraphModel(node_features, hidden_dim=128)
transformer_model = TransformerEncoder(vocab_size, d_model=256)

# 3. Train models
train_ncf!(ncf_model, train_data, val_data; epochs=100)
train_lstm!(lstm_model, train_data; epochs=50)
train_gnn!(gnn_model, knowledge_graph; epochs=100)
train_transformer!(transformer_model, text_data; epochs=50)

# 4. Create ensemble
ensemble = HybridEnsemble(ncf_model, lstm_model, gnn_model, transformer_model)
train_ensemble!(ensemble, train_data; epochs=20)

# 5. Evaluate
results = evaluate_all_models([ncf_model, lstm_model, gnn_model, transformer_model, ensemble], test_data)
generate_latex_table(results)
plot_comparison(results)

# 6. Generate recommendations
recommendations = recommend_courses(ensemble, employee_id=123, k=5)
```

---

## 📝 Notes for Thesis

- Document all hyperparameter choices with justification
- Report training times and computational resources used
- Include ablation studies (e.g., NCF with/without MLP component)
- Compare with baseline methods (random, popularity-based, traditional CF)
- Discuss cold-start handling for new employees
- Address scalability considerations for Telecom Namibia's employee base

---

**Created:** January 10, 2026  
**For:** MSc Thesis - IDP Recommender Model using Deep Learning  
**Author:** Hubert Patrick Mouton
