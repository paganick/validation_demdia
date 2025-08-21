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
    def bert_validate(cls, df, tokenizer, dataset_type='bluesky', include_shap=True, 
                    n_shap_samples=50, n_runs=3, random_seeds=None):
        """
        Fast 3-run BERT validation with comprehensive results storage and SHAP analysis.
        
        Args:
            df: DataFrame with 'text' and 'labels' columns
            tokenizer: Hugging Face tokenizer
            dataset_type: 'bluesky' or 'twitter' - for reference
            include_shap: Whether to perform SHAP analysis
            n_shap_samples: Number of samples for SHAP
            n_runs: Number of training runs (default: 3)
            random_seeds: List of seeds to use. If None, uses [42, 123, 456]
        
        Returns:
            trainer: Best performing trainer (for backward compatibility)
            report: Classification report from best run (for backward compatibility)
            cm: Median confusion matrix across all runs
            shap_data: SHAP analysis results (if include_shap=True)
            shap_summary_stats: SHAP summary statistics (if include_shap=True)
            
        Additional attributes stored in trainer:
            trainer.all_run_results: Detailed results from all runs
            trainer.results_summary: Statistical summary across runs
        """
        import shap
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
            
            # Speed-optimized training arguments with epoch-based evaluation
            if is_small_dataset:
                training_args = TrainingArguments(
                    output_dir=f"./BERT_models/run_{run_idx}",
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
                    save_total_limit=1,
                    seed=seed,
                    data_seed=seed,
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=2,
                    report_to=None,
                    remove_unused_columns=True,
                    dataloader_pin_memory=False,
                )
                early_stopping_patience = 3
                
            else:
                training_args = TrainingArguments(
                    output_dir=f"./BERT_models/run_{run_idx}",
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
                    save_total_limit=1,
                    gradient_accumulation_steps=1,
                    max_grad_norm=1.0,
                    seed=seed,
                    data_seed=seed,
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=4,
                    report_to=None,
                    remove_unused_columns=True,
                    dataloader_pin_memory=torch.cuda.is_available(),
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
        
        # SHAP Analysis Section (integrated from your code)
        shap_data = None
        shap_summary_stats = None
        
        if include_shap:
            print("\nStarting SHAP analysis...")
            
            # Use the best model and its validation data
            model = median_trainer.model
            
            # Get validation data from best run
            val_texts = all_results[median_idx]['val_texts']
            val_labels = all_results[median_idx]['true_labels']
            preds = all_results[median_idx]['predictions']
            random_seed = all_results[median_idx]['seed']
            
            # ---------------------------
            # 1. Prediction wrapper
            # ---------------------------
            def predict_fn(texts):
                # Ensure input is a list of strings
                if isinstance(texts, str):
                    texts = [texts]
                texts = [str(t) for t in texts]

                # Tokenize
                inputs = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )

                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                return probs.cpu().numpy()

            # ---------------------------
            # 2. Deterministic sampling
            # ---------------------------
            n_shap_samples = min(n_shap_samples, len(val_texts))
            np.random.seed(random_seed)
            shap_indices = np.random.choice(len(val_texts), n_shap_samples, replace=False)
            shap_indices = np.sort(shap_indices)

            shap_texts = [str(val_texts[i]) for i in shap_indices]
            shap_true_labels = [val_labels[i] for i in shap_indices]
            shap_pred_labels = [preds[i] for i in shap_indices]

            # ---------------------------
            # 3. Initialize SHAP explainer
            # ---------------------------
            try:
                print(f"Using SHAP version: {shap.__version__}")
                background_size = min(10, len(shap_texts))
                background_texts = shap_texts[:background_size]

                masker = shap.maskers.Text(tokenizer)
                explainer = shap.Explainer(predict_fn, masker)
                print("SHAP explainer initialized successfully.")
            except Exception as e:
                print(f"Failed to create SHAP explainer: {e}")
                shap_data = {'error': str(e), 'dataset_type': dataset_type}
                shap_summary_stats = {'error': str(e), 'dataset_type': dataset_type}

            # ---------------------------
            # 4. Compute SHAP values
            # ---------------------------
            if shap_data is None:  # Only proceed if explainer was created successfully
                try:
                    sample_size = min(20, len(shap_texts))
                    sample_texts = [t.strip() for t in shap_texts[:sample_size] if t.strip()]
                    if not sample_texts:
                        raise ValueError("No valid text samples for SHAP.")

                    print(f"Computing SHAP values for {len(sample_texts)} samples...")
                    shap_values = explainer(sample_texts)
                    print("SHAP values computed successfully.")
                except Exception as e:
                    print(f"Error computing SHAP values: {e}")
                    shap_data = {'error': str(e), 'dataset_type': dataset_type}
                    shap_summary_stats = {'error': str(e), 'dataset_type': dataset_type}

            # ---------------------------
            # 5. Process SHAP results robustly
            # ---------------------------
            if shap_data is None:  # Only proceed if SHAP values were computed successfully
                try:
                    # Case 1: shap_values has .values (ExplainerOutput)
                    if hasattr(shap_values, 'values'):
                        shap_vals = shap_values.values
                        base_vals = getattr(shap_values, 'base_values', None)
                        feature_names = getattr(shap_values, 'feature_names', None)
                    else:
                        shap_vals = shap_values
                        base_vals = None

                    # Ensure shap_vals is always a list of arrays (one per sample)
                    shap_vals_list = []
                    for val in shap_vals:
                        arr = np.array(val)
                        if arr.ndim == 1:
                            arr = arr[:, np.newaxis]  # single-feature case
                        shap_vals_list.append(arr)

                    shap_data = {
                        'shap_values': shap_vals_list,      # list of arrays
                        'base_values': base_vals,
                        'texts': sample_texts,
                        'true_labels': shap_true_labels[:len(sample_texts)],
                        'pred_labels': shap_pred_labels[:len(sample_texts)],
                        'dataset_type': dataset_type,
                        'feature_names': feature_names,
                        'class_names': ['AI', 'human']
                    }

                    # Optional summary (mean over each sample, ignoring raggedness)
                    mean_abs_list = [np.mean(np.abs(x)) for x in shap_vals_list]
                    shap_summary_stats = {
                        'mean_abs_shap': np.mean(mean_abs_list),
                        'dataset_type': dataset_type
                    }

                except Exception as e:
                    print(f"Error processing SHAP results: {e}")
                    print("Returning partial SHAP results...")
                    shap_data = {'error': str(e), 'dataset_type': dataset_type}
                    shap_summary_stats = {'error': str(e), 'dataset_type': dataset_type}
        
        # Return values for backward compatibility
        return median_trainer, median_report, median_cm, all_results, shap_data, shap_summary_stats
    
    @classmethod
    # def save_shap_plotting_data(cls, shap_data, shap_summary_stats, output_path="shap_plotting_data.pkl"):
    #     """
    #     Save SHAP data to a file for external plotting.
    #     """
    #     import pickle
    #     plotting_data = {
    #         'shap_data': shap_data,
    #         'shap_summary_stats': shap_summary_stats,
    #         'metadata': {
    #             'n_samples': len(shap_data['texts']),
    #             'n_classes': len(shap_data['class_names']),
    #             'dataset_type': shap_data.get('dataset_type', 'unknown'),
    #             'creation_timestamp': pd.Timestamp.now().isoformat()
    #         }
    #     }
        
    #     with open(output_path, 'wb') as f:
    #         pickle.dump(plotting_data, f)
        
    #     print(f"SHAP plotting data saved to {output_path}")
    #     return output_path

    @classmethod
    def save_shap_plotting_data(cls, shap_data, shap_summary_stats, output_path="shap_plotting_data.json"):
        """
        Save SHAP data to a JSON file for external plotting.
        """
        # Convert numpy objects to JSON-serializable types
        def to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (list, tuple)):
                return [to_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            else:
                return obj

        serializable_shap_data = to_serializable(shap_data)
        serializable_summary_stats = to_serializable(shap_summary_stats)

        # Ensure class_names exists
        class_names = shap_data.get('class_names', ['class_0'])

        plotting_data = {
            'shap_data': serializable_shap_data,
            'shap_summary_stats': serializable_summary_stats,
            'metadata': {
                'n_samples': len(shap_data.get('texts', [])),
                'n_classes': len(class_names),
                'dataset_type': shap_data.get('dataset_type', 'unknown'),
                'creation_timestamp': pd.Timestamp.now().isoformat()
            }
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(plotting_data, f, indent=2)

        print(f"SHAP plotting data saved to {output_path}")
        return output_path



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

## Help functions for retrieval methods

def build_bm25(history):
        tokenized = [doc.split() for doc in history]
        return BM25Okapi(tokenized)

def retrieve_context(prompt_text, bm25_client, history, k=3):
    query = prompt_text.split()
    top_docs = bm25_client.get_top_n(query, history, n=k)
    return "\n".join(top_docs)

## Help function for training personalized model

def format_conversation(row):
    # Plain prompt without any special tokens.
    return f"You are @{row['username']}. Respond to: '{row['reply_to']}'\n\n{row['message']}"

def preprocess_function(tokenizer, examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized