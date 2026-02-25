# Thesis Chapter Outlines

## IDP Recommender System - 4-Model Architecture (Updated January 2026)

The Individual Development Plan (IDP) Recommender System implements a comprehensive **4-Model Deep Learning Architecture** for personalized employee development recommendations.

| Model | Purpose | Architecture | Primary Metric | Performance |
|-------|---------|--------------|----------------|-------------|
| **Model 1: Skill-Course Recommendation** | Map skill gaps to relevant courses | Two-Stage Hybrid (NCF Scorer + Coverage) | Coverage | **95.6%** |
| **Model 2: Career Path Prediction** | Predict career transitions | Graph Neural Network (GNN) | NDCG@10 | **0.849** |
| **Model 3: Development Action** | Recommend 70-20-10 learning actions | Neural Collaborative Filtering (NCF) | Accuracy | **64.7%** |
| **Model 4: Mentor Matching** | Match mentors to mentees | Direct Matcher (Feature MLP with Skill Overlap) | Accuracy | **76.4%** |

---

## Chapter 5: Implementation

This chapter covers the complete implementation of the IDP Recommender System, from data collection through model training to system integration.

### 5.1 System Architecture

#### 5.1.1 Technology Stack
| Component | Technology | Version |
|-----------|------------|---------|
| Programming Language | Julia | 1.11 |
| Deep Learning Framework | Flux.jl | 0.14+ |
| Data Processing | DataFrames.jl, CSV.jl | Latest |
| Visualization | Plots.jl, StatsPlots.jl | Latest |
| Model Serialization | BSON.jl | Latest |
| API Framework | HTTP.jl, JSON3.jl | Latest |

#### 5.1.2 Hardware Environment
- **CPU**: Intel Core i7 / AMD Ryzen
- **RAM**: 16GB minimum
- **GPU**: CUDA-compatible (optional, CPU training supported)

#### 5.1.3 System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IDP RECOMMENDER SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐  │
│  │   Raw Data   │───►│              Data Pipeline                        │  │
│  │  (CSV Files) │    │  Loading → Cleaning → Feature Extraction         │  │
│  └──────────────┘    └──────────────────────────────────────────────────┘  │
│                                         │                                    │
│                                         ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         4-Model Architecture                          │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │  │
│  │  │Skill-Course │ │Career Path  │ │Dev Action   │ │Mentor Match │    │  │
│  │  │Two-Stage    │ │GNN          │ │NCF          │ │Direct Match │    │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                         │                                    │
│                                         ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         REST API (HTTP.jl)                            │  │
│  │  /recommend/courses  /predict/career  /recommend/actions  /match/mentor│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                         │                                    │
│                                         ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    TrainEase LMS Integration (Laravel)                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Figure 5.1**: System Architecture Diagram → *(Create manually)*

---

### 5.2 Data Collection

#### 5.2.1 Dataset Overview
The system uses multiple datasets organized in the `Data/` directory:

| Dataset File | Records | Purpose | Used By |
|--------------|---------|---------|---------|
| `Online_Courses.csv` | 8,092 | Skill-Course mapping | Model 1: Skill-Course |
| `all_job_post.csv` | ~14,000 | Job-skill relationships | Model 1: Skill-Course |
| `usa_job_posting_dataset.csv` | ~9,000 | Additional job-skills | Model 1: Skill-Course |
| `jobstreet_all_job_dataset.csv` | 13,834 | Job market insights | Job market integration |
| `career_path_in_all_field.csv` | 9,000 | Career transitions | Model 2: Career Path |
| `employee_data.csv` | 300 | Employee profiles | Model 3 & 4 |
| `user_feedback.csv` | 431 | System feedback | Reinforcement learning |

**Dataset Usage Summary**:
- **Model 1 (Skill-Course)**: `all_job_post.csv` + `usa_job_posting_dataset.csv` (appended) + `Online_Courses.csv`
- **Model 2 (Career Path)**: `career_path_in_all_field.csv`
- **Model 3 (Dev Action)**: `employee_data.csv` + `development_actions/`
- **Model 4 (Mentor Match)**: `employee_data.csv` + `mentoring/`
- **Job Market Insights**: `jobstreet_all_job_dataset.csv` (processed via `integrate_jobstreet_data.jl`)

- **Figure 5.2**: Dataset Overview → `fig_5_1_dataset_overview.png`
- **Table 5.1**: Dataset Summary Statistics

#### 5.2.2 Online Courses Dataset (8,092 records)
**Source**: Aggregated online learning platforms

| Column | Type | Description |
|--------|------|-------------|
| Course Name | String | Title of the course |
| Skills | String (comma-separated) | Skills taught by the course |
| Category | String | Course category (e.g., Technology, Business) |
| Difficulty | String | Beginner, Intermediate, Advanced |
| Duration | Numeric | Course length in hours |

- **Figure 5.3**: Course Distribution by Category → `fig_5_2_course_distribution.png`

#### 5.2.3 Job Postings Datasets
Three job posting datasets provide job-skill relationship data:

| Dataset | Records | Purpose |
|---------|---------|---------|
| `all_job_post.csv` | ~14,000 | Primary job-skill mapping |
| `usa_job_posting_dataset.csv` | ~9,000 | Appended for richer skill extraction |
| `jobstreet_all_job_dataset.csv` | 13,834 | Job market insights (Asian market) |

**Combined Usage**: For Model 1 (Skill-Course), `all_job_post.csv` and `usa_job_posting_dataset.csv` are appended to create ~23,000 job postings.

- **Figure 5.4**: Job Posting Distribution → `fig_5_3_job_distribution.png`

#### 5.2.4 Career Paths Dataset (9,000 records)
**Source**: Career development research data

| Column | Type | Description |
|--------|------|-------------|
| Career | String | Career/job title |
| Field | String | Professional field (15 fields) |
| Skill_1 to Skill_15 | Float | Skill importance ratings (1-10) |

**Fields Covered**: Technology, Finance, Healthcare, Education, Engineering, Marketing, Sales, HR, Legal, Operations, Research, Design, Management, Consulting, Other

#### 5.2.5 Employee/Mentoring Data (300 records)
**Source**: Organizational employee data

| Column | Type | Description |
|--------|------|-------------|
| Employee ID | Integer | Unique identifier |
| Name | String | Employee name |
| Department | String | Organizational department |
| Skills | String | Employee's current skills |
| Experience Years | Integer | Years of experience |
| Is Mentor | Boolean | Whether employee is a mentor |

**Derived Statistics**:
- **Mentors**: 136 unique mentors
- **Mentees**: 151 unique mentees
- **Mentor-Mentee Pairs**: 268 relationships (233 completed)

---

### 5.3 Data Preprocessing

#### 5.3.1 Preprocessing Pipeline
The preprocessing pipeline transforms raw CSV data into model-ready training samples:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA PREPROCESSING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: DATA CLEANING                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Load CSV    │───►│Remove Nulls │───►│Remove Dupes │───►│Filter Lang  │  │
│  │ (Raw Data)  │    │& Empty Rows │    │& Invalid    │    │(English)    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  STAGE 2: TEXT NORMALIZATION                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Lowercase   │───►│ Trim White- │───►│ Standardize │───►│ Remove      │  │
│  │ All Text    │    │ space       │    │ Punctuation │    │ Stop Words  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  STAGE 3: FEATURE EXTRACTION                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Parse Skill │───►│ Build Skill │───►│ One-Hot     │───►│ Create      │  │
│  │ Strings     │    │ Vocabulary  │    │ Encode      │    │ Embeddings  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  STAGE 4: PAIR GENERATION                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Generate    │───►│ Negative    │───►│ Shuffle     │───►│ 70/15/15    │  │
│  │ Pos Pairs   │    │ Sampling    │    │ & Balance   │    │ Split       │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Figure 5.5**: Data Preprocessing Pipeline → *(Create manually)*

#### 5.3.2 Preprocessing Steps Detail

| Stage | Step | Description | Applied To |
|-------|------|-------------|------------|
| 1 | Load CSV | Read raw data with type inference | All datasets |
| 1 | Remove Nulls | Drop rows with missing required fields | All datasets |
| 1 | Remove Duplicates | Deduplicate by primary key | Courses, Jobs |
| 1 | Filter Language | Keep English-only content | Courses |
| 2 | Lowercase | Convert all text to lowercase | Skills, Titles |
| 2 | Trim Whitespace | Remove leading/trailing spaces | All text fields |
| 2 | Standardize | Normalize punctuation, special chars | Skills |
| 2 | Remove Stop Words | Remove common words from skill text | Job descriptions |
| 3 | Parse Skills | Split comma-separated skill strings | Courses, Jobs |
| 3 | Build Vocabulary | Create skill ID mapping (1,247 skills) | All models |
| 3 | One-Hot Encode | Convert categories to binary vectors | Departments, Fields |
| 3 | Create Embeddings | Initialize learnable embeddings | NCF, GNN models |
| 4 | Positive Pairs | Create (entity1, entity2, 1) pairs | All models |
| 4 | Negative Sampling | Sample non-matching pairs (4:1 ratio) | All models |
| 4 | Shuffle & Balance | Randomize order, balance classes | Training set |
| 4 | Train/Val/Test Split | 70% / 15% / 15% stratified split | All datasets |

- **Table 5.2**: Data Cleaning Statistics

#### 5.3.3 Training Data Statistics

| Model | Positive Samples | Negative Samples | Train | Validation | Test | Neg:Pos Ratio |
|-------|-----------------|------------------|-------|------------|------|---------------|
| Skill-Course | 12,895 | 51,580 | 45,132 | 9,671 | 9,672 | 4:1 |
| Career Path | 450 | 2,250 | 1,889 | 405 | 406 | 5:1 |
| Dev Action | 2,100 | 8,400 | 7,350 | 1,575 | 1,575 | 4:1 |
| Mentor Match | 233 | 932 | 816 | 175 | 174 | 4:1 |

- **Figure 5.6**: Training Data Statistics → `fig_5_5_training_data_stats.png`
- **Table 5.3**: Data Split Summary

---

### 5.4 Model Implementation

#### 5.4.1 Model 1: Skill-Course Recommendation (Two-Stage Hybrid)

The Skill-Course model uses a **two-stage hybrid approach** that combines deep learning scoring with coverage optimization.

**Stage 1: NCF-Based Skill-Course Scorer**
```
Input: (Skill Embedding, Course Embedding)
       ↓
GMF Branch: Element-wise Product
MLP Branch: Concatenate → Dense(128) → Dense(64)
       ↓
Fusion: Concatenate → Dense(32) → Dense(1, sigmoid)
       ↓
Output: Relevance Score (0-1)
```

**Stage 2: Coverage Optimization (Greedy Set Cover)**
```julia
function recommend_courses(skill_gaps, scorer_model, course_skills)
    uncovered = Set(skill_gaps)
    selected_courses = []
    
    while !isempty(uncovered)
        best_course = argmax(course -> 
            sum(scorer_model(skill, course) for skill in uncovered ∩ course_skills[course]),
            all_courses
        )
        push!(selected_courses, best_course)
        setdiff!(uncovered, course_skills[best_course])
    end
    
    return selected_courses
end
```

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 64 |
| Hidden Dimensions | [128, 64, 32] |
| Dropout | 0.3 |
| Learning Rate | 1e-3 |
| Batch Size | 64 |
| Epochs | 50 |
| Loss Function | MSE (Regression) |

- **Figure 5.7**: Skill-Course Two-Stage Architecture → *(Create manually)*

#### 5.4.2 Model 2: Career Path Prediction (GNN)

The Career Path model uses a **Graph Neural Network** to capture career transition patterns.

**Architecture**:
```
Input: (Source Career ID, Target Career ID)
       ↓
Embedding Layer: Career → 64-dim vector
       ↓
Message Passing Layer 1: Dense(64 → 128, ReLU)
       ↓
Message Passing Layer 2: Dense(128 → 128, ReLU)
       ↓
Link Prediction: Concatenate → Dense(256 → 128) → Dropout(0.4) → Dense(64) → Dense(1, sigmoid)
       ↓
Output: Transition Probability (0-1)
```

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 64 |
| Hidden Dimensions | [128, 128, 64] |
| Dropout | 0.4 |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| Epochs | 50 |
| Loss Function | Binary Cross-Entropy |

**Alternative Architectures Tested**:
- NCF (Neural Collaborative Filtering)
- Transformer (Attention-based)

- **Figure 5.8**: Career Path GNN Architecture → *(Create manually)*

#### 5.4.3 Model 3: Development Action Recommendation (NCF)

The Development Action model implements the **70-20-10 learning framework** using Neural Collaborative Filtering.

**70-20-10 Framework**:
- 70% Experience (on-the-job learning)
- 20% Exposure (mentoring, networking)
- 10% Education (formal training)

**Architecture**:
```
Input: (Employee Skill Profile, Action Embedding)
       ↓
GMF Branch: Element-wise Product
MLP Branch: Concatenate → Dense(128) → Dense(64) → Dense(32)
       ↓
Fusion: Concatenate → Dense(32) → Dense(1, sigmoid)
       ↓
Post-processing: Apply 70-20-10 quota constraints
       ↓
Output: Ranked Actions (7 Experience + 2 Exposure + 1 Education)
```

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 32 |
| Hidden Dimensions | [128, 64, 32] |
| Dropout | 0.3 |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| Epochs | 100 |
| Loss Function | Binary Cross-Entropy |

**Alternative Architectures Tested**:
- Content-Based (Skill-action similarity)
- Hybrid (Weighted combination)

- **Figure 5.9**: Development Action NCF Architecture → *(Create manually)*

#### 5.4.4 Model 4: Mentor Matching (Direct Matcher)

The Mentor Matching model uses a **feature-based MLP with explicit skill overlap features**.

**Architecture**:
```
Mentor Features (233-dim)    Mentee Features (233-dim)
       ↓                            ↓
Mentor MLP: 233→192→96→48    Mentee MLP: 233→192→96→48
       ↓                            ↓
       └──────────┬─────────────────┘
                  ↓
Skill Overlap Features (4-dim) → Dense(24)
                  ↓
         Concatenate All
                  ↓
Output MLP: 120→96→48→1 (with Dropout 0.5)
                  ↓
Output: Match Probability (0-1)
```

**Skill Overlap Features**:
| Feature | Description |
|---------|-------------|
| `skill_overlap_count` | Number of overlapping skills |
| `skill_overlap_ratio` | Overlap count / mentee skill gaps |
| `expertise_match_score` | Weighted match based on mentor expertise |
| `department_match` | Binary: same department or not |

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 48 |
| Hidden Dimensions | [192, 96, 48, 24] |
| Dropout | 0.5 |
| Learning Rate | 3e-4 |
| Batch Size | 8 |
| Weight Decay | 5e-4 |
| Epochs | 300 (early stopped at ~66) |
| Loss Function | Focal Loss (γ=2.0) |
| Label Smoothing | α=0.1 |
| Negative Ratio | 4:1 (Hard negatives) |

**Key Architectural Improvements** (from 47.3% to 76.4% accuracy):
1. Hard Negative Sampling: Sample similar but non-matching mentors
2. Explicit Skill Overlap Features
3. Higher Regularization (Dropout=0.5, Weight Decay=5e-4)
4. Focal Loss for class imbalance
5. Label Smoothing for regularization

- **Figure 5.10**: Mentor Matching Architecture → *(Create manually)*

---

### 5.5 Training Process

#### 5.5.1 Training Pipeline
All models follow a standardized training pipeline:

```julia
function train_model(model, train_data, val_data; epochs=100, patience=10)
    optimizer = Adam(learning_rate)
    best_val_loss = Inf
    patience_counter = 0
    
    for epoch in 1:epochs
        # Training loop
        for batch in train_data
            grads = gradient(() -> loss(model, batch), params(model))
            update!(optimizer, params(model), grads)
        end
        
        # Validation
        val_loss = evaluate(model, val_data)
        
        # Early stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss
            save_model(model)
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                break
            end
        end
    end
end
```

#### 5.5.2 Hyperparameter Summary

| Model | Embed Dim | Hidden Dims | Dropout | Batch Size | LR | Epochs |
|-------|-----------|-------------|---------|------------|-----|--------|
| Skill-Course | 64 | [128, 64, 32] | 0.3 | 64 | 1e-3 | 50 |
| Career Path (GNN) | 64 | [128, 128, 64] | 0.4 | 32 | 1e-3 | 50 |
| Dev Action (NCF) | 32 | [128, 64, 32] | 0.3 | 32 | 1e-3 | 100 |
| Mentor Match | 48 | [192, 96, 48, 24] | 0.5 | 8 | 3e-4 | 300 |

- **Figure 5.11**: Training Configuration Heatmap → `fig_6_11_config_heatmap.png`

---

### 5.6 API Implementation

#### 5.6.1 REST API Endpoints

| Endpoint | Method | Description | Model Used |
|----------|--------|-------------|------------|
| `/api/recommend/courses` | POST | Get course recommendations for skill gaps | Model 1 |
| `/api/predict/career` | POST | Predict next career transitions | Model 2 |
| `/api/recommend/actions` | POST | Get 70-20-10 development actions | Model 3 |
| `/api/match/mentor` | POST | Find matching mentors for employee | Model 4 |
| `/api/health` | GET | API health check | - |

#### 5.6.2 API Request/Response Examples

**Course Recommendation Request**:
```json
POST /api/recommend/courses
{
  "employee_id": 123,
  "skill_gaps": ["python", "machine learning", "data analysis"]
}
```

**Course Recommendation Response**:
```json
{
  "recommended_courses": [
    {"course_id": 45, "name": "Python for Data Science", "coverage": ["python", "data analysis"]},
    {"course_id": 89, "name": "ML Fundamentals", "coverage": ["machine learning"]}
  ],
  "total_coverage": 1.0,
  "efficiency": 0.67
}
```

---

### 5.7 System Integration

#### 5.7.1 TrainEase LMS Integration

The IDP Recommender integrates with TrainEase LMS (Laravel) through:

1. **REST API**: HTTP endpoints for real-time recommendations
2. **Database Sync**: Employee data synchronization
3. **Event Webhooks**: Course completion and feedback events

**Integration Architecture**:
```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   TrainEase     │◄───────►│   Julia API     │◄───────►│   ML Models     │
│   (Laravel)     │  HTTP   │   (HTTP.jl)     │  BSON   │   (Flux.jl)     │
└─────────────────┘         └─────────────────┘         └─────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   MySQL DB      │         │   User Feedback │         │   Model Files   │
│   (Employees)   │         │   (CSV/DB)      │         │   (.bson)       │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

#### 5.7.2 Production Deployment

**Model Files** (saved in `models/` directory):
| Model File | Size | Description |
|------------|------|-------------|
| `skill_course_ncf.bson` | ~2MB | Skill-Course Scorer |
| `career_path/gnn_enhanced.bson` | ~1MB | Career Path GNN |
| `action_recommender/ncf_production.bson` | ~500KB | Development Action NCF |
| `mentor_matching/direct_matcher_v3.bson` | ~1MB | Mentor Direct Matcher |

---

## Chapter 6: Results and Evaluation

This chapter presents the experimental results and evaluation of all four IDP recommender models.

### 6.1 Experimental Setup

#### 6.1.1 Evaluation Metrics

| Metric | Formula | Used By |
|--------|---------|---------|
| **NDCG@K** | Normalized Discounted Cumulative Gain | Career Path |
| **AUC** | Area Under ROC Curve | Career Path |
| **Accuracy** | (TP + TN) / Total | Dev Action, Mentor |
| **Precision** | TP / (TP + FP) | All models |
| **Recall** | TP / (TP + FN) | All models |
| **F1-Score** | 2 × (P × R) / (P + R) | All models |
| **Coverage** | Covered Skills / Total Skills | Skill-Course |
| **MSE/MAE** | Regression error metrics | Skill-Course Stage 1 |
| **R²** | Coefficient of determination | Skill-Course Stage 1 |
| **Correlation** | Pearson correlation | Skill-Course Stage 1 |

- **Table 6.1**: Experimental Environment
- **Table 6.2**: Evaluation Metrics Summary

---

### 6.2 Model 1 Results: Skill-Course Recommendation

#### 6.2.1 Stage 1 Results (Scorer)
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MSE | 0.0048 | 0.0050 | 0.0051 |
| MAE | 0.055 | 0.056 | 0.057 |
| R² | 0.462 | 0.451 | 0.448 |
| Correlation | 0.695 | 0.688 | 0.682 |

#### 6.2.2 Stage 2 Results (Coverage Optimization)
| Metric | Value |
|--------|-------|
| **Coverage** | **95.6%** |
| Avg Courses Selected | 2.0 |
| Efficiency | 47.8% |
| Success Rate | 100% |

#### 6.2.3 Key Insight
Traditional link prediction (NDCG, MRR) is the wrong paradigm for course recommendation. Coverage optimization ensures ALL skill gaps are addressed, not just the top-K ranked courses.

#### 6.2.4 Figures and Tables
- **Figure 6.1**: Skill-Course Two-Stage Hybrid Results → `fig_6_1_skill_course_two_stage.png`
- **Figure 6.2**: Skill-Course Scorer Training Curves → `fig_6_1b_skill_course_training.png`
- **Figure 6.3**: Link Prediction Comparison → `fig_6_2_skill_course_link_prediction.png` *(Reference)*
- **Table 6.3**: Skill-Course Model Performance Summary

---

### 6.3 Model 2 Results: Career Path Prediction

#### 6.3.1 Architecture Comparison
| Model | NDCG@10 | AUC | F1 | Precision | Recall | Accuracy | Epochs |
|-------|---------|-----|-----|-----------|--------|----------|--------|
| **GNN** | **0.849** | 0.634 | **0.245** | **0.480** | **0.164** | 0.781 | 50 |
| NCF | 0.823 | 0.630 | 0.189 | 0.409 | 0.123 | 0.772 | 50 |
| Transformer | 0.827 | **0.692** | 0.0 | 0.0 | 0.0 | **0.784** | 26 |

**Best Model**: GNN (highest NDCG@10=0.849, best F1=0.245)

#### 6.3.2 Training Convergence
- GNN: Converged at epoch 50, stable validation loss
- NCF: Converged at epoch 50, slight overfitting after epoch 40
- Transformer: Early stopped at epoch 26 due to gradient issues

#### 6.3.3 Figures and Tables
- **Figure 6.4**: Career Path Architecture Comparison → `fig_6_3_career_path_comparison.png`
- **Figure 6.5**: Career Path Training History → `fig_6_4_career_path_training.png`
- **Figure 6.6**: Career Path Hyperparameters → `fig_6_5_career_path_hyperparams.png`
- **Table 6.4**: Career Path Model Performance

---

### 6.4 Model 3 Results: Development Action Recommendation

#### 6.4.1 Architecture Comparison
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **NCF** | **0.647** | 0.521 | 0.521 | **0.521** |
| Content-Based | 0.626 | 0.492 | 0.479 | 0.485 |
| Hybrid | 0.620 | 0.485 | 0.521 | 0.502 |

**Best Model**: NCF (highest accuracy=64.7%, F1=0.521)

#### 6.4.2 70-20-10 Distribution Validation
The model correctly recommends actions following the 70-20-10 framework:
- Experience actions: 70% of recommendations
- Exposure actions: 20% of recommendations
- Education actions: 10% of recommendations

#### 6.4.3 Figures and Tables
- **Figure 6.7**: Development Action Comparison → `fig_6_6_action_comparison.png`
- **Figure 6.8**: Action Type Distribution → `fig_6_7_action_distribution.png`
- **Table 6.5**: Action Recommender Performance

---

### 6.5 Model 4 Results: Mentor Matching

#### 6.5.1 Architecture Comparison
| Model | Accuracy | Correlation | Precision | Recall | F1 | NDCG@10 |
|-------|----------|-------------|-----------|--------|-----|---------|
| **Direct Matcher** | **0.764** | 0.168 | 0.250 | 0.161 | 0.196 | **0.345** |
| Enhanced MLP | 0.747 | **0.228** | 0.276 | 0.258 | **0.267** | 0.324 |

**Best Model**: Direct Matcher (highest accuracy=76.4%)

#### 6.5.2 Performance Improvement Analysis
| Version | Accuracy | Key Changes |
|---------|----------|-------------|
| Baseline | 47.3% | Basic NCF |
| v2 | 65.8% | + Hard negative sampling |
| v3 (Final) | 76.4% | + Skill overlap features, Focal loss |

**Total Improvement**: +29.1 percentage points

#### 6.5.3 Figures and Tables
- **Figure 6.9**: Mentor Matching Architecture Comparison → `fig_6_8_mentor_comparison.png`
- **Figure 6.10**: Mentor Matching Detailed Metrics → `fig_6_9_mentor_detailed.png`
- **Figure 6.11**: Mentor Matching Training Curves → `fig_6_9b_mentor_training.png`
- **Table 6.6**: Mentor Matching Performance

---

### 6.6 Overall System Performance

#### 6.6.1 Summary of Best Models

| Component | Selected Model | Primary Metric | Value |
|-----------|---------------|----------------|-------|
| Skill-Course | Two-Stage Hybrid | Coverage | **95.6%** |
| Career Path | GNN | NDCG@10 | **0.849** |
| Development Action | NCF | Accuracy | **64.7%** |
| Mentor Matching | Direct Matcher | Accuracy | **76.4%** |

#### 6.6.2 Figures and Tables
- **Figure 6.12**: All Models Performance Summary → `fig_6_10_all_models_summary.png`
- **Figure 6.13**: Training Configuration Heatmap → `fig_6_11_config_heatmap.png`
- **Table 6.7**: Final Model Selection Summary

---

### 6.7 Discussion

#### 6.7.1 Key Findings
1. **Skill-Course**: Two-stage hybrid with coverage optimization outperforms link prediction paradigm
2. **Career Path**: GNN captures graph structure of career transitions effectively (NDCG=0.849)
3. **Development Action**: NCF provides personalized action recommendations (Acc=64.7%)
4. **Mentor Matching**: Direct Matcher with skill overlap features achieves 76.4% accuracy

#### 6.7.2 Paradigm Shift: Coverage vs Link Prediction
- Traditional recommendation metrics (NDCG, MRR) optimize for ranking
- IDP requires **complete skill gap coverage**, not just top-K ranking
- Two-stage hybrid addresses this fundamental difference

#### 6.7.3 Limitations
1. **Cold Start Problem**: New users without skill profiles have limited recommendations
2. **Data Sparsity**: Limited mentor-mentee pairs (233) affects matching quality
3. **No A/B Testing**: Production validation pending
4. **Single Organization**: Employee data from one organization may limit generalization

---

## Summary of Figures and Tables

### Chapter 5: Implementation
| ID | Type | Title | Filename |
|----|------|-------|----------|
| 5.1 | Figure | System Architecture Diagram | *(Create manually)* |
| 5.2 | Figure | Dataset Overview | `fig_5_1_dataset_overview.png` |
| 5.3 | Figure | Course Distribution by Category | `fig_5_2_course_distribution.png` |
| 5.4 | Figure | Job Posting Distribution | `fig_5_3_job_distribution.png` |
| 5.5 | Figure | Data Preprocessing Pipeline | *(Create manually)* |
| 5.6 | Figure | Training Data Statistics | `fig_5_5_training_data_stats.png` |
| 5.7 | Figure | Skill-Course Two-Stage Architecture | *(Create manually)* |
| 5.8 | Figure | Career Path GNN Architecture | *(Create manually)* |
| 5.9 | Figure | Development Action NCF Architecture | *(Create manually)* |
| 5.10 | Figure | Mentor Matching Architecture | *(Create manually)* |
| 5.11 | Figure | Training Configuration Heatmap | `fig_6_11_config_heatmap.png` |
| 5.1 | Table | Dataset Summary Statistics | - |
| 5.2 | Table | Data Cleaning Statistics | - |
| 5.3 | Table | Data Split Summary | - |

### Chapter 6: Results and Evaluation
| ID | Type | Title | Filename |
|----|------|-------|----------|
| 6.1 | Figure | Skill-Course Two-Stage Results | `fig_6_1_skill_course_two_stage.png` |
| 6.2 | Figure | Skill-Course Training Curves | `fig_6_1b_skill_course_training.png` |
| 6.3 | Figure | Link Prediction Comparison | `fig_6_2_skill_course_link_prediction.png` |
| 6.4 | Figure | Career Path Architecture Comparison | `fig_6_3_career_path_comparison.png` |
| 6.5 | Figure | Career Path Training History | `fig_6_4_career_path_training.png` |
| 6.6 | Figure | Career Path Hyperparameters | `fig_6_5_career_path_hyperparams.png` |
| 6.7 | Figure | Development Action Comparison | `fig_6_6_action_comparison.png` |
| 6.8 | Figure | Action Type Distribution | `fig_6_7_action_distribution.png` |
| 6.9 | Figure | Mentor Matching Comparison | `fig_6_8_mentor_comparison.png` |
| 6.10 | Figure | Mentor Matching Detailed Metrics | `fig_6_9_mentor_detailed.png` |
| 6.11 | Figure | Mentor Matching Training Curves | `fig_6_9b_mentor_training.png` |
| 6.12 | Figure | All Models Performance Summary | `fig_6_10_all_models_summary.png` |
| 6.13 | Figure | Training Configuration Heatmap | `fig_6_11_config_heatmap.png` |
| 6.1 | Table | Experimental Environment | - |
| 6.2 | Table | Evaluation Metrics Summary | - |
| 6.3 | Table | Skill-Course Model Performance | - |
| 6.4 | Table | Career Path Model Performance | - |
| 6.5 | Table | Action Recommender Performance | - |
| 6.6 | Table | Mentor Matching Performance | - |
| 6.7 | Table | Final Model Selection Summary | - |

---

## Key Metrics Summary

### Model 1: Skill-Course Recommendation
- **Architecture**: Two-Stage Hybrid (NCF Scorer + Coverage Optimization)
- **Stage 1 (Scorer)**: MSE=0.0051, MAE=0.057, R²=0.448, Correlation=0.682
- **Stage 2 (Coverage)**: Coverage=95.6%, Courses=2.0, Efficiency=47.8%

### Model 2: Career Path Prediction
- **Architecture**: Graph Neural Network (GNN)
- **Metrics**: NDCG@10=0.849, AUC=0.634, F1=0.245, Accuracy=0.781

### Model 3: Development Action
- **Architecture**: Neural Collaborative Filtering (NCF)
- **Metrics**: Accuracy=0.647, F1=0.521, Precision=0.521, Recall=0.521

### Model 4: Mentor Matching
- **Architecture**: Direct Matcher (Feature MLP with Skill Overlap)
- **Metrics**: Accuracy=0.764, NDCG@10=0.345, F1=0.196
- **Improvement**: 47.3% → 76.4% (+29.1 percentage points)

---

## Generated Thesis Figures

Run `julia scripts/generate_thesis_figures_v2.jl` to generate all figures.

**Output Directory**: `C:\Users\MoutonH\Desktop\Individual_Development_Plan__IDP__Recommender_Model_using_Deep_Learning (1)\figures\Assets`

### Chapter 5: Implementation Figures
| Filename | Description |
|----------|-------------|
| `fig_5_1_dataset_overview.png` | Dataset Overview (4 datasets with record counts) |
| `fig_5_2_course_distribution.png` | Course Distribution by Category |
| `fig_5_3_job_distribution.png` | Job Posting Distribution |
| `fig_5_5_training_data_stats.png` | Training Data Statistics |
| `fig_6_11_config_heatmap.png` | Training Configuration Heatmap |

### Chapter 6: Results Figures
| Filename | Description |
|----------|-------------|
| `fig_6_1_skill_course_two_stage.png` | Skill-Course Two-Stage Hybrid Results |
| `fig_6_1b_skill_course_training.png` | Skill-Course Scorer Training Curves |
| `fig_6_2_skill_course_link_prediction.png` | Link Prediction Comparison (Reference) |
| `fig_6_3_career_path_comparison.png` | Career Path Architecture Comparison |
| `fig_6_4_career_path_training.png` | Career Path Training History |
| `fig_6_5_career_path_hyperparams.png` | Career Path Hyperparameters |
| `fig_6_6_action_comparison.png` | Development Action Comparison |
| `fig_6_7_action_distribution.png` | Action Type Distribution (70-20-10) |
| `fig_6_8_mentor_comparison.png` | Mentor Matching Comparison |
| `fig_6_9_mentor_detailed.png` | Mentor Matching Detailed Metrics |
| `fig_6_9b_mentor_training.png` | Mentor Matching Training Curves |
| `fig_6_10_all_models_summary.png` | All Models Performance Summary |

### Figures to Create Manually
| Figure | Description |
|--------|-------------|
| Figure 5.1 | System Architecture Diagram |
| Figure 5.5 | Data Preprocessing Pipeline (4 Stages) |
| Figure 5.7 | Skill-Course Two-Stage Architecture |
| Figure 5.8 | Career Path GNN Architecture |
| Figure 5.9 | Development Action NCF Architecture |
| Figure 5.10 | Mentor Matching Architecture |

---

## Appendix A: Model Architecture Diagrams

### A.1 Model 1: Two-Stage Hybrid (Skill-Course)
```
Stage 1: NCF-Based Skill-Course Scorer
────────────────────────────────────────
Input: (Skill ID, Course ID)
       ↓
Skill Embedding (64-dim)    Course Embedding (64-dim)
       ↓                            ↓
GMF: Element-wise Product    MLP: Concat → Dense(128) → Dense(64)
       ↓                            ↓
       └──────────┬─────────────────┘
                  ↓
         Fusion: Concat → Dense(32) → Dense(1, sigmoid)
                  ↓
         Output: Relevance Score (0-1)

Stage 2: Coverage Optimization
────────────────────────────────
Input: Skill Gaps, Scorer Model, Course-Skill Mapping
       ↓
While uncovered skills remain:
  - Score all courses for remaining skills
  - Select course with highest aggregate score
  - Mark covered skills as complete
       ↓
Output: Minimum course set with maximum coverage
```

### A.2 Model 2: GNN (Career Path)
```
Input: (Source Career ID, Target Career ID)
       ↓
Career Embedding Layer (64-dim)
       ↓
Message Passing Layer 1: Dense(64 → 128, ReLU)
       ↓
Message Passing Layer 2: Dense(128 → 128, ReLU)
       ↓
Link Prediction Head:
  Concat(src, tgt) → Dense(256→128) → Dropout(0.4) → Dense(64) → Dense(1, sigmoid)
       ↓
Output: Transition Probability (0-1)
```

### A.3 Model 3: NCF (Development Action)
```
Input: (Employee Skill Profile, Action ID)
       ↓
Employee Embedding (32-dim)    Action Embedding (32-dim)
       ↓                            ↓
GMF: Element-wise Product    MLP: Concat → Dense(128→64→32)
       ↓                            ↓
       └──────────┬─────────────────┘
                  ↓
         Fusion: Concat → Dense(32) → Dense(1, sigmoid)
                  ↓
         Post-process: Apply 70-20-10 quotas
                  ↓
         Output: 7 Experience + 2 Exposure + 1 Education
```

### A.4 Model 4: Direct Matcher (Mentor Matching)
```
Mentor Features (233-dim)    Mentee Features (233-dim)
       ↓                            ↓
Mentor MLP: 233→192→96→48    Mentee MLP: 233→192→96→48
       ↓                            ↓
       └──────────┬─────────────────┘
                  │
Skill Overlap Features (4-dim)
  - skill_overlap_count
  - skill_overlap_ratio
  - expertise_match_score
  - department_match
                  ↓
         Overlap MLP: 4→24
                  ↓
         Concat: [mentor_emb, mentee_emb, overlap_emb]
                  ↓
         Output MLP: 120→96→48→1 (Dropout=0.5)
                  ↓
         Output: Match Probability (Focal Loss, γ=2.0)
```

