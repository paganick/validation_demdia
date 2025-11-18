from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from empath import Empath
from scipy.stats import ranksums
from scipy.spatial.distance import euclidean
from rank_bm25 import BM25Okapi
import json
import torch
from datetime import datetime
import uuid
            

class Validator:
    """
    Validator class to perform validation using BERT and Empath.

    TODO: Note: this is missing two types of validation, as I was not yet able to fully automate these. These are available separately.
    - Validation 3: Do the models correctly predict attributes of the individual that they are modeling? E.g., political affiliation. (See llama_finetuning_annotatino.ipnyb)
    - Validation 4: Minimodels. Do the models fulfill some basic behavior of human conversation? 
    """
    
    @classmethod 
    def validate(cls, df):
        """
        Perform both BERT-based and Empath-based validation on the dataset.
        This takes a dataframe with column 'text' and column 'labels'. 
        The latter is an int that specifiees whether it is written by a human or not. 
        """
        print("Running BERT validation...")
        cls.bert_validate(df)
        print("Running EMPATH validation...")
        cls.empath_validate(df)
    
    @classmethod
    def bert_validate(cls, df, tokenizer, include_shap=False,
                    n_shap_samples=500, n_runs=3, random_seeds=None):
        """
        Fast 3-run BERT validation with comprehensive results storage.

        Args:
            df: DataFrame with 'text' and 'labels' columns
            tokenizer: Hugging Face tokenizer
            include_shap: (Deprecated) Kept for backward compatibility, SHAP analysis removed
            n_shap_samples: (Deprecated) Kept for backward compatibility
            n_runs: Number of training runs (default: 3)
            random_seeds: List of seeds to use. If None, uses [42, 123, 456]

        Returns:
            trainer: Best performing trainer (for backward compatibility)
            report: Classification report from best run (for backward compatibility)
            cm: Median confusion matrix across all runs
            all_results: Detailed results from all runs
            shap_data: (Deprecated) Always returns empty dict for backward compatibility
            shap_summary_stats: (Deprecated) Always returns empty dict for backward compatibility

        Additional attributes stored in trainer:
            trainer.all_run_results: Detailed results from all runs
            trainer.results_summary: Statistical summary across runs
        """
        import os
        import random
        from sklearn.utils.class_weight import compute_class_weight
        from transformers import EarlyStoppingCallback
        
        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            try:
                from transformers import set_seed as transformers_set_seed
                transformers_set_seed(seed)
            except ImportError:
                pass
        
        # Generate random seeds if not provided
        if random_seeds is None:
            random_seeds = [42, 123, 456][:n_runs]
        
        print(f"Running {n_runs} training runs with seeds: {random_seeds}")
        
        os.makedirs("./BERT_models", exist_ok=True)
        
        # Check dataset characteristics
        dataset_size = len(df)
        is_small_dataset = dataset_size < 5000
        
        print(f"Dataset size: {dataset_size}")
        print(f"Using {'small' if is_small_dataset else 'standard'} dataset strategy")
        print("Class distribution:")
        print(df['labels'].value_counts())
        
        all_results = []
        all_trainers = []
        all_confusion_matrices = []
        
        for run_idx, seed in enumerate(random_seeds):
            print(f"\n{'='*40}")
            print(f"Run {run_idx + 1}/{n_runs} with seed {seed}")
            print(f"{'='*40}")
            
            set_seed(seed)
            
            # Optimized train/test split with stratification
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                df['text'].tolist(), df['labels'].tolist(), 
                test_size=0.2, random_state=seed, stratify=df['labels'].tolist()
            )
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

            # Create datasets
            train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
            val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})

            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)

            # Load fresh model for each run
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]   # short random string
            pid = os.getpid()                  # process ID

            output_dir = f"./BERT_models/run_{run_idx}_{timestamp}_{pid}_{unique_id}"

            print(f"Output directory: {output_dir}")       

            # Speed-optimized training arguments with epoch-based evaluation
            if is_small_dataset:
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    eval_strategy='epoch',
                    save_strategy='epoch',
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=32,
                    num_train_epochs=6,
                    learning_rate=3e-5,
                    weight_decay=0.1,
                    warmup_ratio=0.1,
                    logging_steps=50,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                    save_total_limit=6,
                    seed=seed,
                    data_seed=seed,
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=2,
                    report_to=None,
                    remove_unused_columns=True,
                    dataloader_pin_memory=False,
                    overwrite_output_dir=True
                )
                early_stopping_patience = 3
                
            else:
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    eval_strategy='epoch',
                    save_strategy='epoch',
                    per_device_train_batch_size=32,
                    per_device_eval_batch_size=64,
                    num_train_epochs=3,
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    warmup_steps=300,
                    logging_steps=100,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                    save_total_limit=3,
                    gradient_accumulation_steps=1,
                    max_grad_norm=1.0,
                    seed=seed,
                    data_seed=seed,
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=4,
                    report_to=None,
                    remove_unused_columns=True,
                    dataloader_pin_memory=torch.cuda.is_available(),
                    overwrite_output_dir=True
                )
                early_stopping_patience = 2

            # Class weighting for balanced training
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(train_labels), 
                y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights)
            
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.get("logits")
                    
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
                    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                    
                    return (loss, outputs) if return_outputs else loss

            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                
                accuracy = accuracy_score(labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
                
                return {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            )

            # Train
            print("Starting training...")
            trainer.train()

            # Predictions
            predictions = trainer.predict(val_dataset)
            preds = np.argmax(predictions.predictions, axis=1)

            # Metrics
            report = classification_report(val_labels, preds, output_dict=True)
            cm = confusion_matrix(val_labels, preds)
            
            # Store detailed results for this run
            run_result = {
                'run': run_idx + 1,
                'seed': seed,
                'accuracy': report['accuracy'],
                'f1_weighted': report['weighted avg']['f1-score'],
                'f1_macro': report['macro avg']['f1-score'],
                'precision_weighted': report['weighted avg']['precision'],
                'recall_weighted': report['weighted avg']['recall'],
                'precision_macro': report['macro avg']['precision'],
                'recall_macro': report['macro avg']['recall'],
                'report': report,
                'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm,
                'predictions': preds.tolist() if isinstance(preds, np.ndarray) else preds,
                'true_labels': val_labels.tolist() if isinstance(val_labels, np.ndarray) else val_labels,
                'val_texts': val_texts,  # Store for potential re-analysis
                'training_history': trainer.state.log_history if hasattr(trainer.state, 'log_history') else None
            }
            import shutil
            shutil.rmtree(output_dir)  # delete checkpoints folder

            all_results.append(run_result)
            all_confusion_matrices.append(cm)
            all_trainers.append(trainer)
            
            print(f"Run {run_idx + 1} Results:")
            print(f"  Accuracy: {report['accuracy']:.4f}")
            print(f"  F1 (weighted): {report['weighted avg']['f1-score']:.4f}")
            print(f"  F1 (macro): {report['macro avg']['f1-score']:.4f}")
            
        # Calculate summary statistics across runs
        metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted', 
                'precision_macro', 'recall_macro']
        results_summary = {}
        
        print(f"\n{'='*50}")
        print(f"SUMMARY ACROSS {n_runs} RUNS")
        print(f"{'='*50}")
        
        for metric in metrics:
            values = [result[metric] for result in all_results]
            results_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values,
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
            }
            
            print(f"{metric.upper()}:")
            print(f"  Mean: {results_summary[metric]['mean']:.4f} ± {results_summary[metric]['std']:.4f}")
            print(f"  Median: {results_summary[metric]['median']:.4f}")
            print(f"  Range: [{results_summary[metric]['min']:.4f}, {results_summary[metric]['max']:.4f}]")
            print(f"  CV: {results_summary[metric]['coefficient_of_variation']:.3f}")
        
        # Calculate confidence intervals (95%)
        print(f"\n95% CONFIDENCE INTERVALS:")
        for metric in ['accuracy', 'f1_weighted']:
            mean = results_summary[metric]['mean']
            std = results_summary[metric]['std']
            ci = 1.96 * std / np.sqrt(n_runs)  # 95% CI
            print(f"{metric.upper()}: {mean:.4f} ± {ci:.4f}")
        
        # ---- NEW PART: Median Trainer ----
        accuracies = [res['accuracy'] for res in all_results]
        median_accuracy = np.median(accuracies)

        # Find run whose accuracy is closest to the median
        median_idx = int(np.argmin([abs(acc - median_accuracy) for acc in accuracies]))
        median_result = all_results[median_idx]

        # Median trainer and associated results
        median_trainer = all_trainers[median_idx]
        median_report = median_result['report']
        median_cm = median_result['confusion_matrix']

        print(f"\nMedian Trainer corresponds to Run {median_result['run']} (Seed={median_result['seed']})")
        print(f"Accuracy: {median_result['accuracy']:.4f}")
        print("Median Confusion Matrix:")
        print(median_cm)

        # SHAP Analysis removed - return empty dicts for backward compatibility
        shap_data = {}
        shap_summary_stats = {}

        if include_shap:
            print("\nWarning: SHAP analysis has been removed from this codebase.")
            print("The include_shap parameter is deprecated and will be ignored.")
        
        # Store additional results in the median trainer for backward compatibility
        median_trainer.all_run_results = all_results
        median_trainer.results_summary = results_summary
        
        # Return values for backward compatibility
        return median_trainer, median_report, median_cm, all_results, shap_data, shap_summary_stats
    

    @classmethod
    def save_shap_plotting_data(cls, shap_data, shap_summary_stats, output_path="shap_plotting_data.json"):
        """
        (Deprecated) SHAP analysis has been removed from this codebase.

        This method is kept for backward compatibility but does nothing.

        Args:
            shap_data: (Deprecated) Ignored
            shap_summary_stats: (Deprecated) Ignored
            output_path: (Deprecated) Ignored

        Returns:
            None
        """
        # No-op for backward compatibility
        return None



    @classmethod
    def empath_validate(cls, df):
        """
        Validate dataset using Empath for linguistic analysis.
        """
        
        lexicon = Empath()  # Initialize Empath lexicon

        def analyze_texts(texts):
            """Analyze text using Empath categories."""
            return [lexicon.analyze(text, normalize=True) for text in texts]

        list1 = list(df.loc[df['labels'] == 0, 'text'].values)
        list2 = list(df.loc[df['labels'] == 1, 'text'].values)

        features_list1 = analyze_texts(list1)
        features_list2 = analyze_texts(list2)

        # Convert to DataFrame for easier manipulation
        df1 = pd.DataFrame(features_list1).fillna(0)
        df2 = pd.DataFrame(features_list2).fillna(0)

        # Perform Wilcoxon rank-sum test
        results = []

        for feature in df1.columns:
            stat, p_value = ranksums(df1[feature], df2[feature])
            avg_list1 = df1[feature].mean()
            avg_list2 = df2[feature].mean()
            delta = avg_list2 - avg_list1
            direction = (
                "more in class 1" if delta > 0 else
                "more in class 0" if delta < 0 else
                "no difference"
            )

            results.append({
                "feature": feature,
                "statistic": stat,
                "p_value": p_value,
                "avg_list1": avg_list1,
                "avg_list2": avg_list2,
                "difference": delta,
                "direction": direction
            })

        # Apply Bonferroni correction
        results_df = pd.DataFrame(results)
        # results_df['adjusted_p_value'] = results_df['p_value'] * len(results_df)

        from statsmodels.stats.multitest import multipletests
        results_df['adjusted_p_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]


        # Identify significant features
        significant_features = results_df[results_df['adjusted_p_value'] < 0.05]

        # Calculate Euclidean distance between the average feature vectors of the two lists
        avg_vector_list1 = df1.mean().values
        avg_vector_list2 = df2.mean().values
        distance = euclidean(avg_vector_list1, avg_vector_list2)

        # Display results
        print("Significant Linguistic Features:")
        print(significant_features.sort_values('adjusted_p_value'))
        print(f"Number significantly different features: {len(significant_features)} out of {len(results_df)}")
        print(f"Euclidean Distance Between Groups: {distance}")

        return significant_features.sort_values('adjusted_p_value'), distance


# Note: Helper functions for retrieval and training (build_bm25, retrieve_context,
# format_conversation, preprocess_function) have been moved to simulation/model_utils.py
# to avoid duplication. Import from there if needed:
#   from simulation.model_utils import build_bm25, retrieve_context, etc.