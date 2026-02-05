import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

def plot_confusion_matrix(X_test, y_test, pipe):
    ticks = np.unique(y_test)
    labels = [('class '+ str(tick)) for tick in ticks]
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


class VectorizerClassifierAnalyzer:
    """
    Class to analyze the performance of different combinations of vectorizers 
    and classifiers, measuring training and prediction time.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the analyzer with training and test data.
        
        Parameters:
        -----------
        X_train : array-like
            Training texts
        X_test : array-like
            Test texts
        y_train : array-like
            Training labels
        y_test : array-like
            Test labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []
    
    def evaluate_single_model(self, vectorizer, classifier, vec_name, clf_name, verbose=True):
        """
        Evaluate a specific combination of vectorizer and classifier.
        
        Parameters:
        -----------
        vectorizer : sklearn vectorizer
            Text vectorizer (CountVectorizer, TfidfVectorizer, etc.)
        classifier : sklearn classifier
            Classifier (LogisticRegression, LinearSVC, etc.)
        vec_name : str
            Descriptive name of the vectorizer
        clf_name : str
            Descriptive name of the classifier
        verbose : bool, default=True
            If True, prints progress
            
        Returns:
        --------
        dict : Dictionary with metrics and timing information
        """
        if verbose:
            print(f"Evaluating: {vec_name} + {clf_name}... ", end="")
        
        # Create pipeline
        pipe = make_pipeline(vectorizer, classifier)
        
        # Measure training time
        start_train = time.time()
        pipe.fit(self.X_train, self.y_train)
        train_time = time.time() - start_train
        
        # Measure prediction time
        start_pred = time.time()
        y_pred = pipe.predict(self.X_test)
        pred_time = time.time() - start_pred
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        macro_f1 = f1_score(self.y_test, y_pred, average="macro")
        
        if verbose:
            print(f"(Train: {train_time:.2f}s, Pred: {pred_time:.3f}s)")
        
        return {
            "Vectorizer": vec_name,
            "Classifier": clf_name,
            "Accuracy": accuracy,
            "Macro-F1": macro_f1,
            "Train Time (s)": train_time,
            "Pred Time (s)": pred_time,
            "Total Time (s)": train_time + pred_time
        }
    
    def evaluate_combinations(self, vectorizers, classifiers, verbose=True):
        """
        Evaluate all combinations of vectorizers and classifiers.
        
        Parameters:
        -----------
        vectorizers : dict
            Dictionary {name: vectorizer}
        classifiers : dict
            Dictionary {name: classifier}
        verbose : bool, default=True
            If True, prints progress
            
        Returns:
        --------
        pd.DataFrame : DataFrame with results sorted by Macro-F1
        """
        self.results = []
        
        if verbose:
            print("="*80)
            print("Starting analysis of classifiers and vectorizers")
            print("="*80)
            print(f"Total combinations: {len(vectorizers)} × {len(classifiers)} = {len(vectorizers) * len(classifiers)}")
            print()
        
        for vec_name, vectorizer in vectorizers.items():
            for clf_name, classifier in classifiers.items():
                try:
                    result = self.evaluate_single_model(
                        vectorizer, classifier, vec_name, clf_name, verbose
                    )
                    self.results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"Error: {str(e)}")
                    self.results.append({
                        "Vectorizer": vec_name,
                        "Classifier": clf_name,
                        "Accuracy": np.nan,
                        "Macro-F1": np.nan,
                        "Train Time (s)": np.nan,
                        "Pred Time (s)": np.nan,
                        "Total Time (s)": np.nan,
                        "Error": str(e)
                    })
        
        # Create DataFrame and sort by Macro-F1
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
        
        if verbose:
            print()
            print("="*80)
            print("Analysis Completed")
            print("="*80)
            print(f"Total time: {results_df['Total Time (s)'].sum():.2f}s")
            print()
        
        return results_df
    
    def get_summary_table(self, results_df=None, top_n=10):
        """
        Generate a summary table with the top N combinations.
        
        Parameters:
        -----------
        results_df : pd.DataFrame, optional
            DataFrame with results. If None, uses self.results
        top_n : int, default=10
            Number of best combinations to display
            
        Returns:
        --------
        pd.DataFrame : Formatted summary table
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results).sort_values("Macro-F1", ascending=False)
        
        summary = results_df.head(top_n).copy()
        
        # Format numeric columns for better visualization (4 decimals)
        summary["Accuracy"] = summary["Accuracy"].apply(lambda x: f"{x:.4f}")
        summary["Macro-F1"] = summary["Macro-F1"].apply(lambda x: f"{x:.4f}")
        summary["Train Time (s)"] = summary["Train Time (s)"].apply(lambda x: f"{x:.2f}")
        summary["Pred Time (s)"] = summary["Pred Time (s)"].apply(lambda x: f"{x:.3f}")
        summary["Total Time (s)"] = summary["Total Time (s)"].apply(lambda x: f"{x:.2f}")
        
        return summary
    
    def plot_performance_comparison(self, results_df=None, figsize=(14, 6)):
        """
        Generate comparative performance visualizations.
        
        Parameters:
        -----------
        results_df : pd.DataFrame, optional
            DataFrame with results. If None, uses self.results
        figsize : tuple, default=(14, 6)
            Figure size
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Macro-F1 vs Training Time
        ax1 = axes[0]
        for clf in results_df['Classifier'].unique():
            data = results_df[results_df['Classifier'] == clf]
            ax1.scatter(data['Train Time (s)'], data['Macro-F1'], 
                       label=clf, s=100, alpha=0.7)
        ax1.set_xlabel('Train Time (s)', fontsize=12)
        ax1.set_ylabel('Macro-F1', fontsize=12)
        ax1.set_title('Macro-F1 vs Training Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Top 10 combinations by Macro-F1
        ax2 = axes[1]
        top_10 = results_df.nlargest(10, 'Macro-F1')
        labels = [f"{row['Vectorizer'][:15]}\n{row['Classifier']}" 
                 for _, row in top_10.iterrows()]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
        bars = ax2.barh(range(len(top_10)), top_10['Macro-F1'], color=colors)
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Macro-F1', fontsize=12)
        ax2.set_title('Top 10 Combinations', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars, top_10['Macro-F1'])):
            ax2.text(val + 0.001, i, f'{val:.4f}', 
                    va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()