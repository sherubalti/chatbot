

# import os, json, pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import normalize
# from sklearn.neighbors import NearestNeighbors
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
# import pandas as pd
# from datetime import datetime

# def create_index():
#     DATA_FILE = "qa.json"
#     INDEX_DIR = "index_data"
#     MODEL_NAME = "all-MiniLM-L6-v2"
#     GRAPHS_DIR = "graphs"

#     os.makedirs(INDEX_DIR, exist_ok=True)
#     os.makedirs(GRAPHS_DIR, exist_ok=True)

#     print("Loading data...")
#     with open(DATA_FILE, "r", encoding="utf-8") as f:
#         docs = json.load(f)

#     questions = [d["question"] for d in docs]
#     answers = [d["answer"] for d in docs]
#     ids = list(range(len(docs)))

#     print(f"Loaded {len(docs)} Q&A items.")

#     print("Loading embedding model:", MODEL_NAME)
#     model = SentenceTransformer(MODEL_NAME)

#     print("Computing embeddings...")
#     embeddings = model.encode(questions, show_progress_bar=True, 
#                             convert_to_numpy=True, batch_size=64)
    
#     # Normalize for cosine similarity
#     embeddings = normalize(embeddings, norm="l2", axis=1)

#     # Save raw embeddings + meta
#     np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
#     with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
#         pickle.dump({"questions": questions, "answers": answers, "ids": ids}, f)

#     # Build NearestNeighbors index
#     print("Building NearestNeighbors index...")
#     nn = NearestNeighbors(n_neighbors=min(8, len(questions)), metric="cosine", n_jobs=-1)
#     nn.fit(embeddings)
#     joblib.dump(nn, os.path.join(INDEX_DIR, "nn.joblib"))

#     print(f"Index built and saved to {INDEX_DIR}")
#     print(f"Embeddings shape: {embeddings.shape}")

#     # Evaluate model accuracy and create graphs
#     print("\nEvaluating model performance...")
#     evaluate_model_performance(questions, embeddings, nn, GRAPHS_DIR)

#     print(f"Graphs saved to {GRAPHS_DIR}")

# def evaluate_model_performance(questions, embeddings, nn, graphs_dir):
#     """Evaluate the model performance and create visualization graphs"""
    
#     # Set style for better looking graphs
#     plt.style.use('default')
#     sns.set_palette("husl")
    
#     # 1. Similarity Distribution Graph
#     print("Creating similarity distribution graph...")
#     create_similarity_distribution(questions, embeddings, nn, graphs_dir)
    
#     # 2. Embedding Visualization (2D PCA)
#     print("Creating embedding visualization...")
#     create_embedding_visualization(embeddings, graphs_dir)
    
#     # 3. Performance Metrics
#     print("Calculating performance metrics...")
#     calculate_performance_metrics(questions, embeddings, nn, graphs_dir)
    
#     # 4. Dataset Statistics
#     print("Creating dataset statistics...")
#     create_dataset_statistics(questions, graphs_dir)

# def create_similarity_distribution(questions, embeddings, nn, graphs_dir):
#     """Create histogram of similarity scores"""
#     plt.figure(figsize=(12, 8))
    
#     # Get similarities for each question to its nearest neighbors (excluding itself)
#     if len(questions) > 1:
#         distances, indices = nn.kneighbors(embeddings, n_neighbors=min(6, len(questions)))
#         similarities = 1 - distances[:, 1:]  # Exclude self-similarity (distance=0)
        
#         # Flatten and filter out invalid values
#         similarities_flat = similarities.flatten()
#         similarities_flat = similarities_flat[~np.isnan(similarities_flat)]
#         similarities_flat = similarities_flat[similarities_flat > 0]
        
#         # Create subplots
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Histogram
#         ax1.hist(similarities_flat, bins=50, alpha=0.7, edgecolor='black')
#         ax1.set_xlabel('Cosine Similarity Score')
#         ax1.set_ylabel('Frequency')
#         ax1.set_title('Distribution of Similarity Scores\n(Nearest Neighbors)')
#         ax1.grid(True, alpha=0.3)
        
#         # Add statistics
#         mean_sim = np.mean(similarities_flat)
#         median_sim = np.median(similarities_flat)
#         ax1.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
#         ax1.axvline(median_sim, color='green', linestyle='--', label=f'Median: {median_sim:.3f}')
#         ax1.legend()
        
#         # Box plot
#         ax2.boxplot(similarities_flat)
#         ax2.set_ylabel('Cosine Similarity Score')
#         ax2.set_title('Similarity Scores Box Plot')
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(graphs_dir, 'similarity_distribution.png'), dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"  - Average similarity: {mean_sim:.3f}")
#         print(f"  - Median similarity: {median_sim:.3f}")
#         print(f"  - Similarity range: {np.min(similarities_flat):.3f} to {np.max(similarities_flat):.3f}")

# def create_embedding_visualization(embeddings, graphs_dir):
#     """Create 2D visualization of embeddings using PCA"""
#     from sklearn.decomposition import PCA
    
#     if len(embeddings) > 10:  # Only if we have enough data points
#         try:
#             # Reduce to 2D using PCA
#             pca = PCA(n_components=2, random_state=42)
#             embeddings_2d = pca.fit_transform(embeddings)
            
#             plt.figure(figsize=(12, 8))
            
#             # Create scatter plot
#             scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
#                                 alpha=0.6, s=30, cmap='viridis')
            
#             plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
#             plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
#             plt.title('2D Visualization of Question Embeddings (PCA)')
#             plt.colorbar(scatter, label='Density')
#             plt.grid(True, alpha=0.3)
            
#             # Add explained variance information
#             total_variance = pca.explained_variance_ratio_.sum()
#             plt.figtext(0.02, 0.02, f'Total explained variance: {total_variance:.2%}', 
#                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
#             plt.tight_layout()
#             plt.savefig(os.path.join(graphs_dir, 'embedding_visualization.png'), dpi=300, bbox_inches='tight')
#             plt.close()
            
#             print(f"  - PCA explained variance: {total_variance:.2%}")
            
#         except Exception as e:
#             print(f"  - PCA visualization failed: {e}")

# def calculate_performance_metrics(questions, embeddings, nn, graphs_dir):
#     """Calculate and visualize performance metrics"""
    
#     if len(questions) > 20:  # Only for reasonably sized datasets
#         try:
#             # Simulate retrieval performance
#             n_test = min(50, len(questions) // 4)
#             test_indices = np.random.choice(len(questions), n_test, replace=False)
            
#             precisions = []
#             recalls = []
            
#             for idx in test_indices:
#                 query_embedding = embeddings[idx].reshape(1, -1)
#                 distances, indices = nn.kneighbors(query_embedding, n_neighbors=min(11, len(questions)))
                
#                 # Consider top 5 results
#                 top_k = min(5, len(indices[0]) - 1)
#                 if top_k > 0:
#                     # Simple metric: count how many of the top results have high similarity
#                     similarities = 1 - distances[0][1:top_k+1]  # Exclude self
#                     precision = np.mean(similarities > 0.7)  # Threshold for good matches
#                     recall = len(similarities[similarities > 0.7]) / len(similarities)
                    
#                     precisions.append(precision)
#                     recalls.append(recall)
            
#             if precisions and recalls:
#                 avg_precision = np.mean(precisions)
#                 avg_recall = np.mean(recalls)
                
#                 # Create performance metrics visualization
#                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
#                 # Precision-Recall scatter
#                 ax1.scatter(precisions, recalls, alpha=0.6, color='blue')
#                 ax1.set_xlabel('Precision')
#                 ax1.set_ylabel('Recall')
#                 ax1.set_title(f'Precision vs Recall (Avg: P={avg_precision:.3f}, R={avg_recall:.3f})')
#                 ax1.grid(True, alpha=0.3)
#                 ax1.set_xlim(0, 1)
#                 ax1.set_ylim(0, 1)
                
#                 # Metrics distribution
#                 metrics_data = [precisions, recalls]
#                 ax2.boxplot(metrics_data, labels=['Precision', 'Recall'])
#                 ax2.set_ylabel('Score')
#                 ax2.set_title('Performance Metrics Distribution')
#                 ax2.grid(True, alpha=0.3)
                
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(graphs_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
#                 plt.close()
                
#                 print(f"  - Average Precision: {avg_precision:.3f}")
#                 print(f"  - Average Recall: {avg_recall:.3f}")
                
#         except Exception as e:
#             print(f"  - Performance metrics calculation failed: {e}")

# def create_dataset_statistics(questions, graphs_dir):
#     """Create visualizations for dataset statistics"""
    
#     # Question length analysis
#     question_lengths = [len(q.split()) for q in questions]
    
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
#     # Question length distribution
#     ax1.hist(question_lengths, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
#     ax1.set_xlabel('Number of Words')
#     ax1.set_ylabel('Frequency')
#     ax1.set_title('Distribution of Question Lengths')
#     ax1.grid(True, alpha=0.3)
    
#     # Add statistics
#     avg_length = np.mean(question_lengths)
#     ax1.axvline(avg_length, color='red', linestyle='--', label=f'Average: {avg_length:.1f} words')
#     ax1.legend()
    
#     # Word count box plot
#     ax2.boxplot(question_lengths)
#     ax2.set_ylabel('Number of Words')
#     ax2.set_title('Question Lengths Box Plot')
#     ax2.grid(True, alpha=0.3)
    
#     # Dataset size info
#     total_questions = len(questions)
#     unique_questions = len(set(questions))
#     duplicate_rate = (total_questions - unique_questions) / total_questions if total_questions > 0 else 0
    
#     ax3.text(0.1, 0.7, f'Total Questions: {total_questions}', fontsize=14, fontweight='bold')
#     ax3.text(0.1, 0.5, f'Unique Questions: {unique_questions}', fontsize=14, fontweight='bold')
#     ax3.text(0.1, 0.3, f'Duplicate Rate: {duplicate_rate:.2%}', fontsize=14, fontweight='bold')
#     ax3.text(0.1, 0.1, f'Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=12)
#     ax3.set_xlim(0, 1)
#     ax3.set_ylim(0, 1)
#     ax3.set_title('Dataset Summary', fontsize=16, fontweight='bold')
#     ax3.axis('off')
    
#     # Question length statistics
#     length_stats = {
#         'Min': np.min(question_lengths),
#         'Max': np.max(question_lengths),
#         'Mean': np.mean(question_lengths),
#         'Median': np.median(question_lengths),
#         'Std': np.std(question_lengths)
#     }
    
#     stats_text = '\n'.join([f'{k}: {v:.1f}' for k, v in length_stats.items()])
#     ax4.text(0.1, 0.7, 'Question Length Statistics:', fontsize=14, fontweight='bold')
#     ax4.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace')
#     ax4.set_xlim(0, 1)
#     ax4.set_ylim(0, 1)
#     ax4.set_title('Length Statistics', fontsize=16, fontweight='bold')
#     ax4.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(graphs_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"  - Average question length: {avg_length:.1f} words")
#     print(f"  - Total questions: {total_questions}")
#     print(f"  - Unique questions: {unique_questions}")
#     print(f"  - Duplicate rate: {duplicate_rate:.2%}")

# def generate_report(graphs_dir):
#     """Generate a summary report text file"""
#     report_path = os.path.join(graphs_dir, 'model_report.txt')
    
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write("College Chatbot Model Evaluation Report\n")
#         f.write("=" * 50 + "\n\n")
#         f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
#         f.write("Generated Graphs:\n")
#         f.write("- similarity_distribution.png: Distribution of cosine similarity scores\n")
#         f.write("- embedding_visualization.png: 2D PCA visualization of embeddings\n")
#         f.write("- performance_metrics.png: Precision and recall metrics\n")
#         f.write("- dataset_statistics.png: Dataset overview and statistics\n\n")
        
#         f.write("Graphs Location:\n")
#         f.write(f"- Directory: {graphs_dir}\n")
#         f.write("- All graphs are saved as high-resolution PNG files (300 DPI)\n\n")
        
#         f.write("Next Steps:\n")
#         f.write("1. Review similarity distribution for optimal threshold setting\n")
#         f.write("2. Check embedding visualization for clustering patterns\n")
#         f.write("3. Monitor performance metrics for model improvements\n")
#         f.write("4. Use dataset statistics for data quality assessment\n")

# if __name__ == "__main__":
#     create_index() 


import os, json, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
import pandas as pd
from datetime import datetime

def create_index():
    DATA_FILE = "qa.json"
    INDEX_DIR = "index_data"
    MODEL_NAME = "all-MiniLM-L6-v2"
    GRAPHS_DIR = "graphs"

    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print("Loading data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    questions = [d["question"] for d in docs]
    answers = [d["answer"] for d in docs]
    ids = list(range(len(docs)))

    # Build vocabulary for spell correction from both questions and answers
    vocab = set()
    for d in docs:
        for text in [d["question"], d["answer"]]:
            for word in text.lower().split():
                cleaned = word.strip('.,?!:;')
                if len(cleaned) > 2:
                    vocab.add(cleaned)

    print(f"Loaded {len(docs)} Q&A items with {len(vocab)} unique words in vocab.")

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Computing embeddings...")
    embeddings = model.encode(questions, show_progress_bar=True, 
                            convert_to_numpy=True, batch_size=64)
    
    # Normalize for cosine similarity
    embeddings = normalize(embeddings, norm="l2", axis=1)

    # Save raw embeddings + meta
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"questions": questions, "answers": answers, "ids": ids, "vocab": list(vocab)}, f)

    # Build NearestNeighbors index
    print("Building NearestNeighbors index...")
    nn = NearestNeighbors(n_neighbors=min(8, len(questions)), metric="cosine", n_jobs=-1)
    nn.fit(embeddings)
    joblib.dump(nn, os.path.join(INDEX_DIR, "nn.joblib"))

    print(f"Index built and saved to {INDEX_DIR}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Evaluate model accuracy and create graphs
    print("\nEvaluating model performance...")
    evaluate_model_performance(questions, embeddings, nn, GRAPHS_DIR)

    print(f"Graphs saved to {GRAPHS_DIR}")

def evaluate_model_performance(questions, embeddings, nn, graphs_dir):
    """Evaluate the model performance and create visualization graphs"""
    
    # Set style for better looking graphs
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Similarity Distribution Graph
    print("Creating similarity distribution graph...")
    create_similarity_distribution(questions, embeddings, nn, graphs_dir)
    
    # 2. Embedding Visualization (2D PCA)
    print("Creating embedding visualization...")
    create_embedding_visualization(embeddings, graphs_dir)
    
    # 3. Performance Metrics
    print("Calculating performance metrics...")
    calculate_performance_metrics(questions, embeddings, nn, graphs_dir)
    
    # 4. Dataset Statistics
    print("Creating dataset statistics...")
    create_dataset_statistics(questions, graphs_dir)

def create_similarity_distribution(questions, embeddings, nn, graphs_dir):
    """Create histogram of similarity scores"""
    plt.figure(figsize=(12, 8))
    
    # Get similarities for each question to its nearest neighbors (excluding itself)
    if len(questions) > 1:
        distances, indices = nn.kneighbors(embeddings, n_neighbors=min(6, len(questions)))
        similarities = 1 - distances[:, 1:]  # Exclude self-similarity (distance=0)
        
        # Flatten and filter out invalid values
        similarities_flat = similarities.flatten()
        similarities_flat = similarities_flat[~np.isnan(similarities_flat)]
        similarities_flat = similarities_flat[similarities_flat > 0]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(similarities_flat, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Cosine Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Similarity Scores\n(Nearest Neighbors)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_sim = np.mean(similarities_flat)
        median_sim = np.median(similarities_flat)
        ax1.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
        ax1.axvline(median_sim, color='green', linestyle='--', label=f'Median: {median_sim:.3f}')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(similarities_flat)
        ax2.set_ylabel('Cosine Similarity Score')
        ax2.set_title('Similarity Scores Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, 'similarity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Average similarity: {mean_sim:.3f}")
        print(f"  - Median similarity: {median_sim:.3f}")
        print(f"  - Similarity range: {np.min(similarities_flat):.3f} to {np.max(similarities_flat):.3f}")

def create_embedding_visualization(embeddings, graphs_dir):
    """Create 2D visualization of embeddings using PCA"""
    from sklearn.decomposition import PCA
    
    if len(embeddings) > 10:  # Only if we have enough data points
        try:
            # Reduce to 2D using PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                alpha=0.6, s=30, cmap='viridis')
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('2D Visualization of Question Embeddings (PCA)')
            plt.colorbar(scatter, label='Density')
            plt.grid(True, alpha=0.3)
            
            # Add explained variance information
            total_variance = pca.explained_variance_ratio_.sum()
            plt.figtext(0.02, 0.02, f'Total explained variance: {total_variance:.2%}', 
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, 'embedding_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  - PCA explained variance: {total_variance:.2%}")
            
        except Exception as e:
            print(f"  - PCA visualization failed: {e}")

def calculate_performance_metrics(questions, embeddings, nn, graphs_dir):
    """Calculate and visualize performance metrics"""
    
    if len(questions) > 20:  # Only for reasonably sized datasets
        try:
            # Simulate retrieval performance
            n_test = min(50, len(questions) // 4)
            test_indices = np.random.choice(len(questions), n_test, replace=False)
            
            precisions = []
            recalls = []
            
            for idx in test_indices:
                query_embedding = embeddings[idx].reshape(1, -1)
                distances, indices = nn.kneighbors(query_embedding, n_neighbors=min(11, len(questions)))
                
                # Consider top 5 results
                top_k = min(5, len(indices[0]) - 1)
                if top_k > 0:
                    # Simple metric: count how many of the top results have high similarity
                    similarities = 1 - distances[0][1:top_k+1]  # Exclude self
                    precision = np.mean(similarities > 0.7)  # Threshold for good matches
                    recall = len(similarities[similarities > 0.7]) / len(similarities)
                    
                    precisions.append(precision)
                    recalls.append(recall)
            
            if precisions and recalls:
                avg_precision = np.mean(precisions)
                avg_recall = np.mean(recalls)
                
                # Create performance metrics visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Precision-Recall scatter
                ax1.scatter(precisions, recalls, alpha=0.6, color='blue')
                ax1.set_xlabel('Precision')
                ax1.set_ylabel('Recall')
                ax1.set_title(f'Precision vs Recall (Avg: P={avg_precision:.3f}, R={avg_recall:.3f})')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                
                # Metrics distribution
                metrics_data = [precisions, recalls]
                ax2.boxplot(metrics_data, labels=['Precision', 'Recall'])
                ax2.set_ylabel('Score')
                ax2.set_title('Performance Metrics Distribution')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  - Average Precision: {avg_precision:.3f}")
                print(f"  - Average Recall: {avg_recall:.3f}")
                
        except Exception as e:
            print(f"  - Performance metrics calculation failed: {e}")

def create_dataset_statistics(questions, graphs_dir):
    """Create visualizations for dataset statistics"""
    
    # Question length analysis
    question_lengths = [len(q.split()) for q in questions]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Question length distribution
    ax1.hist(question_lengths, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Question Lengths')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    avg_length = np.mean(question_lengths)
    ax1.axvline(avg_length, color='red', linestyle='--', label=f'Average: {avg_length:.1f} words')
    ax1.legend()
    
    # Word count box plot
    ax2.boxplot(question_lengths)
    ax2.set_ylabel('Number of Words')
    ax2.set_title('Question Lengths Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Dataset size info
    total_questions = len(questions)
    unique_questions = len(set(questions))
    duplicate_rate = (total_questions - unique_questions) / total_questions if total_questions > 0 else 0
    
    ax3.text(0.1, 0.7, f'Total Questions: {total_questions}', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.5, f'Unique Questions: {unique_questions}', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.3, f'Duplicate Rate: {duplicate_rate:.2%}', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.1, f'Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Dataset Summary', fontsize=16, fontweight='bold')
    ax3.axis('off')
    
    # Question length statistics
    length_stats = {
        'Min': np.min(question_lengths),
        'Max': np.max(question_lengths),
        'Mean': np.mean(question_lengths),
        'Median': np.median(question_lengths),
        'Std': np.std(question_lengths)
    }
    
    stats_text = '\n'.join([f'{k}: {v:.1f}' for k, v in length_stats.items()])
    ax4.text(0.1, 0.7, 'Question Length Statistics:', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Length Statistics', fontsize=16, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - Average question length: {avg_length:.1f} words")
    print(f"  - Total questions: {total_questions}")
    print(f"  - Unique questions: {unique_questions}")
    print(f"  - Duplicate rate: {duplicate_rate:.2%}")

def generate_report(graphs_dir):
    """Generate a summary report text file"""
    report_path = os.path.join(graphs_dir, 'model_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("College Chatbot Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Generated Graphs:\n")
        f.write("- similarity_distribution.png: Distribution of cosine similarity scores\n")
        f.write("- embedding_visualization.png: 2D PCA visualization of embeddings\n")
        f.write("- performance_metrics.png: Precision and recall metrics\n")
        f.write("- dataset_statistics.png: Dataset overview and statistics\n\n")
        
        f.write("Graphs Location:\n")
        f.write(f"- Directory: {graphs_dir}\n")
        f.write("- All graphs are saved as high-resolution PNG files (300 DPI)\n\n")
        
        f.write("Next Steps:\n")
        f.write("1. Review similarity distribution for optimal threshold setting\n")
        f.write("2. Check embedding visualization for clustering patterns\n")
        f.write("3. Monitor performance metrics for model improvements\n")
        f.write("4. Use dataset statistics for data quality assessment\n")

if __name__ == "__main__":
    create_index()